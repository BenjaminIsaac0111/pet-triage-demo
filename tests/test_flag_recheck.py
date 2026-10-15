"""A flag may only be lifted if every re-check sample agrees, and failures must count against it."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from triage.flag_recheck import agreement_per_flag, mark_unclear, recheck_lifted_flags
from triage.schemas import RedFlagCheck, build_assessment_schema


def check(flag_id, status, quote="a b"):
    return RedFlagCheck(flag_id=flag_id, status=status, evidence_quote=quote)


class TestAgreementPerFlag:
    def test_all_samples_agree(self):
        samples = [[check("breathing", "absent")]] * 3
        assert agreement_per_flag(["breathing"], samples) == {"breathing": 1.0}

    def test_partial_agreement(self):
        samples = [[check("breathing", "absent")], [check("breathing", "present")]]
        assert agreement_per_flag(["breathing"], samples) == {"breathing": 0.5}

    def test_sample_that_skips_the_flag_counts_as_disagreeing(self):
        samples = [[check("breathing", "absent")], []]
        assert agreement_per_flag(["breathing"], samples) == {"breathing": 0.5}


def test_mark_unclear_only_touches_named_flags():
    checks = [check("breathing", "absent"), check("collapse", "absent")]
    updated = mark_unclear(checks, {"breathing"})
    assert [c.status for c in updated] == ["unclear", "absent"]


def sampling_chain(statuses):
    """A stand-in chain that answers with each status in turn ("broken" fails validation)."""
    schema = build_assessment_schema(["breathing"])
    answers = iter(statuses)

    def answer(_inputs):
        status = next(answers)
        flag_status = "absent" if status == "broken" else status
        return schema(
            red_flag_checks=[
                {"flag_id": "breathing", "status": flag_status, "evidence_quote": "a b"}
            ],
            owner_summary="s",
            complaint_code="NOT-A-CODE" if status == "broken" else "DEMO-012",
            requires_photo_upload=False,
            urgency_reason="r",
            urgency_tier="ROUTINE",
        )

    return RunnableLambda(answer)


def recheck(statuses):
    return recheck_lifted_flags(
        sampling_chain(statuses),
        history=[HumanMessage(content="He's not struggling to breathe")],
        lifted_flag_ids=["breathing"],
        sample_count=len(statuses),
    )


def test_unanimous_samples_allow_the_lift():
    result = recheck(["absent", "absent", "absent"])
    assert result.doubted_flags == set()
    assert result.metrics["lift_check_min_agreement"] == 1.0


def test_one_dissenting_sample_keeps_the_flag():
    result = recheck(["absent", "unclear", "absent"])
    assert result.doubted_flags == {"breathing"}
    assert result.agreement == {"breathing": pytest.approx(2 / 3)}


def test_broken_replies_keep_the_flag_and_are_counted():
    result = recheck(["absent", "broken", "broken"])
    assert result.doubted_flags == {"breathing"}
    assert result.metrics["lift_check_failures"] == 2
