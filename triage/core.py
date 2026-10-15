"""The two-turn triage conversation, independent of any UI or tracking.

Turn 1: the owner describes the problem.
  - If the keyword scan finds an emergency phrase, we skip the questions and
    assess straight away. If the emergency is confirmed, the owner is told to
    contact a vet now; nobody should answer quiz questions during a bloat.
  - Otherwise the LLM asks 2-3 follow-up questions.
Turn 2: the owner answers, and the LLM fills the triage record.
"""

import time
from dataclasses import dataclass, field, replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from triage import red_flags
from triage.chains import (
    ModelConfig,
    build_llm,
    build_question_chain,
    build_triage_chain,
    invoke_with_retry,
)
from triage.reference_data import load_complaints
from triage.schemas import TriageRecord

EMERGENCY_MESSAGE = (
    "This could be an emergency. Please contact your vet or nearest emergency vet now."
)


@dataclass
class TurnResult:
    """What one turn produced. Exactly one of questions / record is set."""

    questions: list[str] | None = None
    record: TriageRecord | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class TriageSession:
    def __init__(
        self,
        config: ModelConfig | None = None,
        llm: BaseChatModel | None = None,
        sampling_llm: BaseChatModel | None = None,
    ):
        self.config = config or ModelConfig()
        self.llm = llm or build_llm(self.config)
        # Same model at a higher temperature, used only to double-check a flag
        # before dismissing it.
        self.sampling_llm = sampling_llm or (
            llm or build_llm(replace(self.config, temperature=self.config.lift_check_temperature))
        )
        self.history: list[BaseMessage] = []
        self.questions_asked: list[str] = []

    @property
    def owner_text(self) -> str:
        return "\n".join(m.content for m in self.history if isinstance(m, HumanMessage))

    def start(self, owner_input: str) -> TurnResult:
        self.history = [HumanMessage(content=owner_input)]
        self.questions_asked = []

        if red_flags.scan(owner_input):
            result = self._assess(skipped_questions=True)
            if result.record.urgency_tier == "EMERGENCY":
                return result
            # The flag was lifted with evidence, so carry on as normal. Keep
            # the early assessment's numbers under their own names.
            assess_metrics = {f"early_{k}": v for k, v in result.metrics.items()}
        else:
            assess_metrics = {}

        chain = build_question_chain(self.llm, self.config.prompt_version)
        t0 = time.perf_counter()
        parsed, retries = invoke_with_retry(chain, self.history)
        latency = time.perf_counter() - t0

        self.questions_asked = parsed.clarifying_questions
        self.history.append(AIMessage(content="\n".join(self.questions_asked)))
        metrics = {
            "questions_latency_sec": latency,
            "questions_retries": retries,
            "questions_count": len(self.questions_asked),
            **assess_metrics,
        }
        return TurnResult(questions=self.questions_asked, metrics=metrics)

    def answer(self, owner_answers: str) -> TurnResult:
        self.history.append(HumanMessage(content=owner_answers))
        return self._assess(skipped_questions=False)

    def _assess(self, skipped_questions: bool) -> TurnResult:
        detected = red_flags.scan(self.owner_text)
        chain, _schema = build_triage_chain(self.llm, self.config.prompt_version, detected)

        t0 = time.perf_counter()
        parsed, retries = invoke_with_retry(chain, self.history)
        latency = time.perf_counter() - t0

        checks = list(parsed.red_flag_checks)
        floor = red_flags.apply_floor(parsed.urgency_tier, detected, checks, self.owner_text)

        agreement: dict[str, float] = {}
        check_metrics: dict[str, float] = {}
        if floor.flags_lifted and self.config.lift_check_samples > 0:
            agreement, check_metrics = self._double_check_lifts(detected, floor.flags_lifted)
            doubted = {fid for fid, share in agreement.items() if share < 1.0}
            if doubted:
                checks = red_flags.mark_unclear(checks, doubted)
                floor = red_flags.apply_floor(
                    parsed.urgency_tier, detected, checks, self.owner_text
                )

        record = TriageRecord(
            complaint_code=parsed.complaint_code,
            complaint_term=load_complaints()[parsed.complaint_code].term,
            urgency_tier=floor.urgency_tier,
            model_urgency_tier=parsed.urgency_tier,
            safety_floor_applied=floor.floor_applied,
            red_flags_detected=detected,
            red_flag_checks=[c.model_dump() for c in checks],
            flag_lift_agreement=agreement,
            requires_photo_upload=parsed.requires_photo_upload,
            owner_summary=parsed.owner_summary,
            urgency_reason=parsed.urgency_reason,
            questions_asked=self.questions_asked,
            skipped_questions_for_emergency=skipped_questions and floor.urgency_tier == "EMERGENCY",
        )
        metrics = {
            "triage_latency_sec": latency,
            "triage_retries": retries,
            "red_flags_detected": len(detected),
            "red_flags_lifted": len(floor.flags_lifted),
            "safety_floor_applied": int(floor.floor_applied),
            **check_metrics,
        }
        return TurnResult(record=record, metrics=metrics)

    def _double_check_lifts(
        self, detected: list[str], lifted: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Ask the model again at a higher temperature; dismiss a flag only if all samples agree.

        Only runs when a flag is about to be dismissed, which is rare, so the
        extra calls are not paid on most conversations.
        """
        chain, _schema = build_triage_chain(self.sampling_llm, self.config.prompt_version, detected)
        samples = []
        failures = 0
        t0 = time.perf_counter()
        for _ in range(self.config.lift_check_samples):
            try:
                parsed, _retries = invoke_with_retry(chain, self.history, max_retries=0)
                samples.append(parsed.red_flag_checks)
            except Exception:
                # A broken reply or timeout counts as disagreeing, so the flag is kept.
                # Counted, because otherwise a broken model looks like an unsure one.
                samples.append([])
                failures += 1
        agreement = red_flags.lift_agreement(lifted, samples)
        metrics = {
            "lift_check_latency_sec": time.perf_counter() - t0,
            "lift_check_min_agreement": min(agreement.values()),
            "flags_kept_after_lift_check": sum(share < 1.0 for share in agreement.values()),
            "lift_check_failures": failures,
        }
        return agreement, metrics
