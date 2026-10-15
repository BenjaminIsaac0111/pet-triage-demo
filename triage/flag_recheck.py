"""Re-ask the model before dismissing ("lifting") a red flag.

Lifting a flag is the one decision that could hide an emergency, so before it
happens we ask the same question again a few times at a higher temperature.
The flag is lifted only if every answer agrees the sign is absent. Otherwise
it's marked "unclear", and the safety floor keeps it.

This only runs when a flag is about to be lifted, which is rare, so most
conversations don't pay for the extra calls.

Agreement shows how sure the model is, not whether it's right: a model can be
consistently wrong.
"""

import time
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from triage.chains import invoke_with_retry
from triage.schemas import RedFlagCheck


@dataclass
class RecheckResult:
    # Per flag, the share of samples that also said "absent" (1.0 = all agreed).
    agreement: dict[str, float]
    # Flags where at least one sample disagreed; these must be kept.
    doubted_flags: set[str]
    # Logged to MLflow. Names are kept stable so old runs stay comparable.
    metrics: dict[str, float] = field(default_factory=dict)


def recheck_lifted_flags(
    sampling_chain: Runnable,
    history: list[BaseMessage],
    lifted_flag_ids: list[str],
    sample_count: int,
) -> RecheckResult:
    """Ask the assessment again `sample_count` times and measure agreement per flag.

    Args:
        sampling_chain: the assessment chain, built with a higher-temperature model.
        history: the conversation so far, exactly as the first assessment saw it.
        lifted_flag_ids: the flags the first assessment wanted to dismiss.
        sample_count: how many extra answers to collect.
    """
    # TODO: Run the samples in parallel (e.g. chain.batch). They're independent,
    # and running them one after another took a case from about 22 s to 86 s.
    # TODO: Once vets have labelled real cases, calibrate this decision with
    # conformal risk control, to set and check a limit on wrongly dismissed
    # emergencies, instead of "every sample must agree".
    samples: list[list[RedFlagCheck]] = []
    failures = 0
    started = time.perf_counter()
    for _ in range(sample_count):
        try:
            assessment, _retries = invoke_with_retry(sampling_chain, history, max_retries=0)
            samples.append(assessment.red_flag_checks)
        except Exception:
            # A broken reply or a timeout counts as disagreeing, so the flag is
            # kept. It's counted separately so a broken model doesn't look
            # like an unsure one.
            samples.append([])
            failures += 1

    agreement = agreement_per_flag(lifted_flag_ids, samples)
    doubted = {flag_id for flag_id, share in agreement.items() if share < 1.0}
    return RecheckResult(
        agreement=agreement,
        doubted_flags=doubted,
        metrics={
            "lift_check_latency_sec": time.perf_counter() - started,
            "lift_check_min_agreement": min(agreement.values()),
            "flags_kept_after_lift_check": len(doubted),
            "lift_check_failures": failures,
        },
    )


def agreement_per_flag(flag_ids: list[str], samples: list[list[RedFlagCheck]]) -> dict[str, float]:
    """For each flag, the share of samples that also said it was absent.

    A sample that doesn't mention the flag at all counts as disagreeing.
    """
    agreement = {}
    for flag_id in flag_ids:
        votes_absent = [
            any(check.flag_id == flag_id and check.status == "absent" for check in sample)
            for sample in samples
        ]
        agreement[flag_id] = sum(votes_absent) / len(votes_absent) if votes_absent else 1.0
    return agreement


def mark_unclear(checks: list[RedFlagCheck], flag_ids: set[str]) -> list[RedFlagCheck]:
    """Downgrade the given flags to "unclear", so the safety floor keeps them."""
    return [
        check.model_copy(update={"status": "unclear"}) if check.flag_id in flag_ids else check
        for check in checks
    ]
