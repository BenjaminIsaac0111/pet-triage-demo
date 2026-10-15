"""The shapes of data the system passes around.

Two kinds:
- What the model must fill in: `FollowUpQuestions` (turn 1) and the
  assessment (turn 2), whose schema is built at runtime so the complaint code
  can only be one of the codes in our list.
- What our own code produces: `TriageRecord`, the record handed to the vet.
  It adds fields that code can work out reliably, instead of asking the model.

Field names in `TriageRecord` are saved in MLflow, so renaming them would
break comparisons with older runs.
"""

from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

from triage.reference_data import load_complaints, load_red_flags

UrgencyTier = Literal["EMERGENCY", "URGENT_24H", "ROUTINE"]

# Lets code compare tiers: a higher number is more urgent.
URGENCY_RANK: dict[str, int] = {"ROUTINE": 0, "URGENT_24H": 1, "EMERGENCY": 2}

# present = happening now or recently; absent = the owner clearly says it isn't
# happening; unclear = can't tell. Doubt is treated as present (see safety_floor.py).
FlagStatus = Literal["present", "absent", "unclear"]


def most_urgent(*tiers: str) -> str:
    """Return the most urgent of the given tiers."""
    return max(tiers, key=URGENCY_RANK.__getitem__)


class FollowUpQuestions(BaseModel):
    """Turn 1: what the assistant asks the owner."""

    clarifying_questions: list[str] = Field(
        description="2 or 3 short, kind, plain-English questions for the owner.",
        min_length=1,
        max_length=3,
    )


class RedFlagCheck(BaseModel):
    """The model's reading of one red flag that the keyword check found."""

    flag_id: str = Field(description="The id of the red flag being checked.")
    evidence_quote: str = Field(
        description="The owner's exact words that show whether this sign is happening."
    )
    status: FlagStatus = Field(
        description="present = happening now or recently; absent = the owner clearly says it "
        "is not happening; unclear = cannot tell."
    )


def build_assessment_schema(detected_flag_ids: list[str] | None = None) -> type[BaseModel]:
    """Build the turn-2 schema, with the allowed values baked in.

    The complaint codes (and red-flag ids) become enums. With structured output,
    Ollama enforces the schema while the model writes, so the model can't invent
    a code. It can still pick a real code that's wrong; the evaluation measures that.

    Args:
        detected_flag_ids: the red flags the keyword check found. Limiting
            `flag_id` to these stops the model answering about flags nobody asked about.
            None allows every flag.

    Returns:
        A Pydantic model class to pass to `with_structured_output`.
    """
    allowed_codes = tuple(load_complaints())
    allowed_flag_ids = tuple(detected_flag_ids or load_red_flags())

    check_schema = create_model(
        "RedFlagCheckForThisMessage",
        __base__=RedFlagCheck,
        flag_id=(Literal[allowed_flag_ids], Field(description="The id of the red flag.")),
    )

    # Field order matters: the model writes fields in this order, so the
    # evidence and summary come before the decisions that should rest on them.
    return create_model(
        "TriageAssessment",
        red_flag_checks=(
            list[check_schema],
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
            Literal[allowed_codes],
            Field(description="The single closest presenting complaint code from the list."),
        ),
        requires_photo_upload=(
            bool,
            Field(description="True only if the problem is visible on the outside of the body."),
        ),
        urgency_reason=(str, Field(description="One sentence explaining the urgency choice.")),
        urgency_tier=(UrgencyTier, Field(description="EMERGENCY, URGENT_24H or ROUTINE.")),
    )


class TriageRecord(BaseModel):
    """The final record for the vet. Built by our code from the model's assessment."""

    complaint_code: str
    complaint_term: str  # looked up from the code, not written by the model
    urgency_tier: UrgencyTier  # the final urgency, after the safety floor
    model_urgency_tier: UrgencyTier  # what the model said, kept to see when the floor stepped in
    safety_floor_applied: bool
    red_flags_detected: list[str]
    red_flag_checks: list[RedFlagCheck]
    flag_lift_agreement: dict[str, float] = Field(
        default_factory=dict,
        description="For flags the model wanted to dismiss: share of re-check samples that agreed.",
    )
    requires_photo_upload: bool
    owner_summary: str
    urgency_reason: str
    questions_asked: list[str]
    skipped_questions_for_emergency: bool

    @field_validator("complaint_code")
    @classmethod
    def code_must_exist(cls, value: str) -> str:
        # Backstop for providers that don't enforce the schema while generating.
        if value not in load_complaints():
            raise ValueError(f"Unknown complaint code: {value}")
        return value
