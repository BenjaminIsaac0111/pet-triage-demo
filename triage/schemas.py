"""The shapes of data the system passes around.

Two kinds:
- What the model must fill in: `FollowUpQuestions` (turn 1; v3 used
  `ScopedFollowUpQuestions`) and the assessment (turn 2), whose schema is
  built at runtime so the complaint code can only be one of the codes in our
  list. `ReplyFormat` says which optional fields each prompt version uses.
- What our own code produces: `TriageRecord`, the record handed to the vet.
  It adds fields that code can work out reliably, instead of asking the model.

Field names in `TriageRecord` are saved in MLflow, so renaming them would
break comparisons with older runs.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

from triage.reference_data import load_complaints, load_red_flags

UrgencyTier = Literal["EMERGENCY", "URGENT_24H", "ROUTINE"]

# Lets code compare tiers: a higher number is more urgent.
URGENCY_RANK: dict[str, int] = {"ROUTINE": 0, "URGENT_24H": 1, "EMERGENCY": 2}

# present = happening now or recently; absent = the owner clearly says it isn't
# happening; unclear = can't tell. Doubt is treated as present (see safety_floor.py).
FlagStatus = Literal["present", "absent", "unclear"]


# Which animals the service covers. A product decision, not a model one: the
# service this demo is modelled on focuses on cats and dogs. From prompt v4
# the owner picks the animal, so the model no longer decides it.
Species = Literal["dog", "cat", "other", "not stated"]
SUPPORTED_SPECIES = ("dog", "cat", "not stated")  # "not stated": ask, don't refuse
NOT_A_CAT_OR_DOG = "This service can only help with cats and dogs."

ScopeCheck = Literal["none", "model", "owner"]


@dataclass(frozen=True)
class ReplyFormat:
    """What a prompt version asks the model to fill in.

    Prompts and reply formats are versioned together, so older prompt
    versions still run exactly as written (see chains.reply_format).
    """

    # Who decides whether to triage at all:
    #   "none"  (v1, v2): nobody; everything is triaged.
    #   "model" (v3): the model says the species and whether it's a health problem.
    #   "owner" (v4): the owner picks cat, dog or another animal, and code
    #           declines other animals without calling the model.
    scope_check: ScopeCheck = "owner"
    # Tell the model to use DEMO-000 "Other / not listed" when no code fits (v3 on).
    other_code_hint: bool = True
    # Up to v3 the model decides whether a photo would help; from v4 code
    # decides from the complaint category (see `photo_would_help`).
    model_sets_photo_flag: bool = False


LATEST_REPLY_FORMAT = ReplyFormat()

# Used by prompt v3 only, and placed first so the model decides scope before
# it writes anything else.
SPECIES_FIELD = Field(
    description="The animal the owner is asking about: dog, cat, other (any other animal, "
    "including wild animals) or not stated."
)
HEALTH_CONCERN_FIELD = Field(
    description="True if the owner describes a possible health or welfare problem with "
    "their animal, including a change in behaviour that could be medical (e.g. eating "
    "stones, sudden aggression). False if they want something else, e.g. help with "
    "training, keeping an animal out of the house, or general chat."
)


def reply_species(reply: BaseModel) -> str:
    """The species a reply gives, or "not stated" for a schema without the scope check."""
    return getattr(reply, "species", "not stated")


def is_in_scope(reply: BaseModel) -> bool:
    """Whether a model reply describes something this service should triage.

    Replies from schemas without the scope check are always in scope.
    """
    return reply_species(reply) in SUPPORTED_SPECIES and getattr(reply, "is_health_concern", True)


def out_of_scope_reason(reply: BaseModel) -> str:
    """A short, owner-facing reason for not triaging. Only call when out of scope."""
    if reply_species(reply) not in SUPPORTED_SPECIES:
        return NOT_A_CAT_OR_DOG
    return "This doesn't sound like a health problem we can assess."


def most_urgent(*tiers: str) -> str:
    """Return the most urgent of the given tiers."""
    return max(tiers, key=URGENCY_RANK.__getitem__)


# Complaint categories a vet can judge from a photo: problems on the outside of the body.
PHOTO_CATEGORIES = {"Skin", "Trauma", "Eye"}


def photo_would_help(complaint_code: str, urgency_tier: str) -> bool:
    """Whether to ask the owner for a photo. Decided by code from prompt v4 on.

    Never in an emergency: the owner should be on the way to a vet, not taking photos.
    """
    category = load_complaints()[complaint_code].category
    return category in PHOTO_CATEGORIES and urgency_tier != "EMERGENCY"


QUESTIONS_FIELD = Field(
    description="2 or 3 short, kind, plain-English questions for the owner.",
    min_length=1,
    max_length=3,
)


class FollowUpQuestions(BaseModel):
    """Turn 1: what the assistant asks the owner (every prompt version except v3)."""

    clarifying_questions: list[str] = QUESTIONS_FIELD


class ScopedFollowUpQuestions(BaseModel):
    """Turn 1 for prompt v3: decide whether to triage at all, then ask the questions."""

    species: Species = SPECIES_FIELD
    is_health_concern: bool = HEALTH_CONCERN_FIELD
    clarifying_questions: list[str] = QUESTIONS_FIELD


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


def build_assessment_schema(
    detected_flag_ids: list[str] | None = None, reply_format: ReplyFormat = LATEST_REPLY_FORMAT
) -> type[BaseModel]:
    """Build the turn-2 schema, with the allowed values baked in.

    The complaint codes (and red-flag ids) become enums. With structured output,
    Ollama enforces the schema while the model writes, so the model can't invent
    a code. It can still pick a real code that's wrong; the evaluation measures that.

    Args:
        detected_flag_ids: the red flags the keyword check found. Limiting
            `flag_id` to these stops the model answering about flags nobody asked about.
            None allows every flag.
        reply_format: which optional fields to include. Older prompt versions
            run with the fields they were written for (see `ReplyFormat`).

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

    scope_fields = (
        {"species": (Species, SPECIES_FIELD), "is_health_concern": (bool, HEALTH_CONCERN_FIELD)}
        if reply_format.scope_check == "model"
        else {}
    )
    code_description = "The single closest presenting complaint code from the list" + (
        ", or DEMO-000 if none of them fits." if reply_format.other_code_hint else "."
    )
    photo_fields = (
        {
            "requires_photo_upload": (
                bool,
                Field(
                    description="True only if the problem is visible on the outside of the body."
                ),
            )
        }
        if reply_format.model_sets_photo_flag
        else {}
    )

    # Field order matters: the model writes fields in this order, so the
    # evidence and summary come before the decisions that should rest on them.
    return create_model(
        "TriageAssessment",
        **scope_fields,
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
            Field(description=code_description),
        ),
        **photo_fields,
        urgency_reason=(str, Field(description="One sentence explaining the urgency choice.")),
        urgency_tier=(UrgencyTier, Field(description="EMERGENCY, URGENT_24H or ROUTINE.")),
    )


class TriageRecord(BaseModel):
    """The final record for the vet. Built by our code from the model's assessment."""

    complaint_code: str
    complaint_term: str  # looked up from the code, not written by the model
    # The owner's pick from v4 (the model's answer in v3). Defaulted so records
    # from older runs still load.
    species: Species = "not stated"
    urgency_tier: UrgencyTier  # the final urgency, after the safety floor
    model_urgency_tier: UrgencyTier  # what the model said, kept to see when the floor stepped in
    safety_floor_applied: bool
    red_flags_detected: list[str]
    red_flag_checks: list[RedFlagCheck]
    flag_lift_agreement: dict[str, float] = Field(
        default_factory=dict,
        description="For flags the model wanted to dismiss: share of re-check samples that agreed.",
    )
    requires_photo_upload: bool  # from v4, decided by `photo_would_help`, not the model
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
