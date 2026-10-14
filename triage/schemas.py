"""Data contracts.

Two kinds of model:
- What the LLM fills in (FollowUpQuestions, and a TriageAssessment built at
  runtime so the complaint code is limited to the codes in our reference file).
- What our own code produces (TriageRecord), which adds fields we can work out
  deterministically instead of trusting the LLM with them.
"""

from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

from triage.reference_data import load_complaints, load_red_flags

UrgencyTier = Literal["EMERGENCY", "URGENT_24H", "ROUTINE"]
URGENCY_RANK: dict[str, int] = {"ROUTINE": 0, "URGENT_24H": 1, "EMERGENCY": 2}

FlagStatus = Literal["present", "absent", "unclear"]


class FollowUpQuestions(BaseModel):
    """Turn 1: what the assistant asks the owner."""

    clarifying_questions: list[str] = Field(
        description="2 or 3 short, kind, plain-English questions for the owner.",
        min_length=1,
        max_length=3,
    )


class RedFlagCheck(BaseModel):
    """The LLM's reading of one red flag that the keyword scan found."""

    flag_id: str = Field(description="The id of the red flag being checked.")
    evidence_quote: str = Field(
        description="The owner's exact words that show whether this sign is happening."
    )
    status: FlagStatus = Field(
        description="present = happening now or recently; absent = the owner clearly says it "
        "is not happening; unclear = cannot tell."
    )


def build_assessment_model(
    complaint_codes: list[str] | None = None, flag_ids: list[str] | None = None
) -> type[BaseModel]:
    """Build the turn-2 schema with the allowed codes baked in as an enum.

    With langchain-ollama's json_schema mode the enum is enforced while the
    model generates, so it cannot invent a code. It can still pick a valid but
    wrong one; that is what the golden-case evaluation measures.

    Field order matters: the model writes fields in order, so the evidence and
    summary come before the decision fields.
    """
    complaints = load_complaints()
    codes = complaint_codes or list(complaints)
    flags = flag_ids or list(load_red_flags())

    check_model = create_model(
        "RedFlagCheckForRun",
        __base__=RedFlagCheck,
        flag_id=(Literal[tuple(flags)], Field(description="The id of the red flag being checked.")),
    )

    model = create_model(
        "TriageAssessment",
        red_flag_checks=(
            list[check_model],
            Field(
                default_factory=list,
                description="One entry per red flag listed in the instructions. "
                "Empty if none were listed.",
            ),
        ),
        owner_summary=(
            str,
            Field(
                description="2-4 plain sentences summarising what the owner reported, for the vet."
            ),
        ),
        complaint_code=(
            Literal[tuple(codes)],
            Field(description="The single closest presenting complaint code from the list."),
        ),
        requires_photo_upload=(
            bool,
            Field(description="True only if the problem is visible on the outside of the body."),
        ),
        urgency_reason=(str, Field(description="One sentence explaining the urgency choice.")),
        urgency_tier=(UrgencyTier, Field(description="EMERGENCY, URGENT_24H or ROUTINE.")),
    )
    return model


class TriageRecord(BaseModel):
    """The final record handed to the vet. Built by our code, not the LLM."""

    complaint_code: str
    complaint_term: str
    urgency_tier: UrgencyTier
    model_urgency_tier: UrgencyTier
    safety_floor_applied: bool
    red_flags_detected: list[str]
    red_flag_checks: list[RedFlagCheck]
    flag_lift_agreement: dict[str, float] = Field(
        default_factory=dict,
        description="For flags the model wanted to dismiss: share of extra samples that agreed.",
    )
    requires_photo_upload: bool
    owner_summary: str
    urgency_reason: str
    questions_asked: list[str]
    skipped_questions_for_emergency: bool

    @field_validator("complaint_code")
    @classmethod
    def code_must_exist(cls, value: str) -> str:
        # Backstop for providers that do not enforce the enum while generating.
        if value not in load_complaints():
            raise ValueError(f"Unknown complaint code: {value}")
        return value
