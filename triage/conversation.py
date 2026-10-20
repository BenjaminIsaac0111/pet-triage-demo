"""The two-turn triage conversation, independent of any UI or tracking.

Turn 1, `start`: the owner picks cat, dog or another animal, and describes the problem.
  - Another animal is declined by code, without calling the model. The
    keyword check still runs: an emergency phrase gets "contact a vet now".
  - If the keyword check finds an emergency phrase, the questions are skipped
    and the assessment runs straight away. If the emergency is confirmed, the
    owner should be told to contact a vet now: nobody should answer questions
    while their dog has bloat.
  - Otherwise the model asks 2-3 follow-up questions.
Turn 2, `answer`: the owner answers, and the model fills the triage record.

Scope is the owner's choice from prompt v4, so a cat or dog is never turned
away. Prompt v3 let the model decide instead (species, and whether it's a
health problem), which could decline real problems; it's kept so the two
can be compared.

Each turn returns a `TurnResult` whose `metrics` (timings, retries, safety
checks) the callers log to MLflow: web_demo.py, terminal_chat.py, evaluate.py.
"""

import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from triage.chains import build_assessment_chain, build_questions_chain, invoke_with_retry
from triage.flag_recheck import RecheckResult, mark_unclear, recheck_lifted_flags
from triage.reference_data import load_complaints
from triage.safety_floor import SafetyFloorResult, apply_safety_floor, find_red_flags
from triage.schemas import (
    NOT_A_CAT_OR_DOG,
    RedFlagCheck,
    TriageRecord,
    out_of_scope_reason,
    photo_would_help,
    reply_format_for,
    reply_species,
)
from triage.settings import TriageSettings

EMERGENCY_MESSAGE = (
    "This could be an emergency. Please contact your vet or nearest emergency vet now."
)
OUT_OF_SCOPE_MESSAGE = (
    "Sorry, we can't triage this. If you're worried about an animal's health, please contact a vet."
)


@dataclass
class TurnResult:
    """What one turn produced. Exactly one of questions, record or out_of_scope_reason is set."""

    questions: list[str] | None = None
    record: TriageRecord | None = None
    out_of_scope_reason: str | None = None  # owner-facing; shown with OUT_OF_SCOPE_MESSAGE
    # Out of scope, but an emergency phrase was found: show EMERGENCY_MESSAGE instead.
    emergency_referral: bool = False
    metrics: dict[str, float] = field(default_factory=dict)  # logged to MLflow by the caller


@dataclass
class _SafetyOutcome:
    """The result of applying the safety rules to one assessment."""

    flag_checks: list[RedFlagCheck]  # possibly downgraded to "unclear" by the re-check
    floor: SafetyFloorResult
    recheck: RecheckResult | None  # None if no flag was about to be lifted


def _call_and_time(chain: Runnable, history: list[BaseMessage]) -> tuple[BaseModel, int, float]:
    """Run a chain with one retry. Returns the reply, the retries used and the seconds taken."""
    started = time.perf_counter()
    reply, retries = invoke_with_retry(chain, history)
    return reply, retries, time.perf_counter() - started


def _out_of_scope(reason: str, metrics: dict[str, float]) -> TurnResult:
    """v3 only: the model's scope answers end the conversation without triaging."""
    supported = int(reason != NOT_A_CAT_OR_DOG)
    return TurnResult(
        out_of_scope_reason=reason,
        metrics={**metrics, "out_of_scope": 1, "species_supported": supported},
    )


class TriageConversation:
    """One owner's conversation, from the first message to the record for the vet.

    Example:
        conversation = TriageConversation(TriageSettings())
        turn = conversation.start("My dog keeps being sick", species="dog")
        if turn.questions:  # the usual path: questions first, then the record
            turn = conversation.answer("Since this morning, but he's still eating")
        print(turn.record)
    """

    def __init__(
        self, settings: TriageSettings | None = None, chat_model: BaseChatModel | None = None
    ):
        """
        Args:
            settings: model and behaviour settings. Read from the environment if omitted.
            chat_model: the model to talk to. Normally built from `settings`; tests
                pass a stand-in, which is then used for the flag re-check too.
        """
        self.settings = settings or TriageSettings()
        self.chat_model = chat_model or self.settings.chat_model()
        self.recheck_chat_model = chat_model or self.settings.chat_model(
            temperature=self.settings.lift_check_temperature
        )
        self.reply_format = reply_format_for(self.settings.prompt_version)
        self.history: list[BaseMessage] = []  # every message so far, owner and assistant
        self.questions_asked: list[str] = []
        self.species = "not stated"  # what the owner picked

    @property
    def all_owner_text(self) -> str:
        """Everything the owner has written. Keywords and quotes are checked against this."""
        return "\n".join(m.content for m in self.history if isinstance(m, HumanMessage))

    # --- The two turns ---

    def start(self, owner_message: str, species: str | None = None) -> TurnResult:
        """Turn 1. Returns follow-up questions, or the final record if an emergency is confirmed.

        Args:
            owner_message: the owner's description of the problem.
            species: what the owner picked: "cat", "dog" or "other". From
                prompt v4 "other" is declined by code. None means not stated.
        """
        # TODO: Allow another round of questions when answers are vague or
        # contradictory. Today there is exactly one round.
        if species not in (None, "cat", "dog", "other"):
            raise ValueError(f"species must be cat, dog or other, not {species!r}")
        self.history = [HumanMessage(content=owner_message)]
        self.questions_asked = []
        self.species = species or "not stated"
        flag_ids = find_red_flags(owner_message)

        if self.species == "other" and self.reply_format.scope_check == "owner":
            # Declined by code, with no model call. Safety beats scope: a rabbit
            # that can't breathe gets "contact a vet now" rather than a plain decline.
            return TurnResult(
                out_of_scope_reason=NOT_A_CAT_OR_DOG,
                emergency_referral=bool(flag_ids),
                metrics={
                    "out_of_scope": 1,
                    "species_supported": 0,
                    "red_flags_detected": len(flag_ids),
                    "emergency_referral": int(bool(flag_ids)),
                },
            )

        early_metrics = {}
        if flag_ids:
            early = self._assess(skipped_questions=True)
            if early.record is None or early.record.urgency_tier == "EMERGENCY":
                return early  # a confirmed emergency, or out of scope
            # The flag was lifted with evidence, so ask questions as usual. The
            # early assessment's numbers are logged under their own names.
            early_metrics = {f"early_{name}": value for name, value in early.metrics.items()}

        chain = build_questions_chain(self.chat_model, self.settings.prompt_version, self.species)
        reply, retries, latency = _call_and_time(chain, self.history)
        metrics = {"questions_latency_sec": latency, "questions_retries": retries, **early_metrics}
        if reason := out_of_scope_reason(reply):  # v3 only: other replies are always in scope
            return _out_of_scope(reason, metrics)

        self.questions_asked = reply.clarifying_questions
        self.history.append(AIMessage(content="\n".join(self.questions_asked)))
        metrics["questions_count"] = len(self.questions_asked)
        return TurnResult(questions=self.questions_asked, metrics=metrics)

    def answer(self, owner_answers: str) -> TurnResult:
        """Turn 2. Returns the final record."""
        self.history.append(HumanMessage(content=owner_answers))
        return self._assess(skipped_questions=False)

    # --- The steps behind each turn ---

    def _assess(self, skipped_questions: bool) -> TurnResult:
        """Ask the model for its assessment, apply the safety rules, and build the record."""
        owner_text = self.all_owner_text
        flag_ids = find_red_flags(owner_text)
        chain = build_assessment_chain(
            self.chat_model, self.settings.prompt_version, flag_ids, self.species
        )
        assessment, retries, latency = _call_and_time(chain, self.history)

        safety = self._apply_safety_rules(assessment, flag_ids, owner_text)
        record = self._build_record(assessment, flag_ids, safety, skipped_questions)
        metrics = {
            # Metric names are kept stable so old MLflow runs stay comparable.
            "triage_latency_sec": latency,
            "triage_retries": retries,
            "red_flags_detected": len(flag_ids),
            "red_flags_lifted": len(safety.floor.flags_lifted),
            "safety_floor_applied": int(safety.floor.floor_raised_urgency),
            **(safety.recheck.metrics if safety.recheck else {}),
        }
        # v3 only: the model's scope answers can end the conversation. Safety beats
        # scope, so an emergency in any animal still gets "contact a vet now".
        reason = out_of_scope_reason(assessment)
        if reason and record.urgency_tier != "EMERGENCY":
            return _out_of_scope(reason, metrics)
        return TurnResult(record=record, metrics=metrics)

    def _apply_safety_rules(
        self, assessment: BaseModel, flag_ids: list[str], owner_text: str
    ) -> _SafetyOutcome:
        """Apply the keyword floor, re-checking any flag the model wants to lift."""
        flag_checks = list(assessment.red_flag_checks)
        floor = apply_safety_floor(assessment.urgency_tier, flag_ids, flag_checks, owner_text)
        if not floor.flags_lifted or self.settings.lift_check_samples <= 0:
            return _SafetyOutcome(flag_checks, floor, recheck=None)

        # TODO: A flag lifted on turn 1 is re-checked again on turn 2, because the
        # owner's first message is still in the text. Skip the second re-check
        # unless the answers mention the sign again.
        recheck = recheck_lifted_flags(
            sampling_chain=build_assessment_chain(
                self.recheck_chat_model, self.settings.prompt_version, flag_ids, self.species
            ),
            history=self.history,
            lifted_flag_ids=floor.flags_lifted,
            sample_count=self.settings.lift_check_samples,
        )
        if recheck.doubted_flags:
            # Any disagreement means the flag can't be lifted: apply the floor again.
            flag_checks = mark_unclear(flag_checks, recheck.doubted_flags)
            floor = apply_safety_floor(assessment.urgency_tier, flag_ids, flag_checks, owner_text)
        return _SafetyOutcome(flag_checks, floor, recheck)

    def _build_record(
        self,
        assessment: BaseModel,
        flag_ids: list[str],
        safety: _SafetyOutcome,
        skipped_questions: bool,
    ) -> TriageRecord:
        """Combine the model's assessment with what our code worked out."""
        final_urgency = safety.floor.urgency_tier
        if self.reply_format.model_sets_photo_flag:  # up to v3
            requires_photo = assessment.requires_photo_upload
        else:
            requires_photo = photo_would_help(assessment.complaint_code, final_urgency)
        return TriageRecord(
            complaint_code=assessment.complaint_code,
            complaint_term=load_complaints()[assessment.complaint_code].term,
            # The owner's pick wins; v3's model answer is used only if they didn't pick.
            species=self.species if self.species != "not stated" else reply_species(assessment),
            urgency_tier=final_urgency,
            model_urgency_tier=assessment.urgency_tier,
            safety_floor_applied=safety.floor.floor_raised_urgency,
            red_flags_detected=flag_ids,
            red_flag_checks=[check.model_dump() for check in safety.flag_checks],
            flag_lift_agreement=safety.recheck.agreement if safety.recheck else {},
            requires_photo_upload=requires_photo,
            owner_summary=assessment.owner_summary,
            urgency_reason=assessment.urgency_reason,
            questions_asked=self.questions_asked,
            skipped_questions_for_emergency=skipped_questions and final_urgency == "EMERGENCY",
        )
