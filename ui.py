"""Simple web demo: the owner's chat on the left, the vet's record on the right.

Usage: streamlit run ui.py

The UI is deliberately thin. All triage logic lives in triage/core.py and is
shared with cli.py and evaluate.py. Each conversation is one MLflow run in the
"pet-triage-ui" experiment.
"""

import io
from pathlib import Path

import mlflow
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from triage import tracking
from triage.chains import PROMPTS_DIR, ModelConfig
from triage.core import EMERGENCY_MESSAGE, TriageSession, TurnResult
from triage.reference_data import load_golden_cases

load_dotenv()
st.set_page_config(page_title="Pet Triage Demo", page_icon="🐾", layout="wide")

URGENCY_DISPLAY = {
    "EMERGENCY": (st.error, "EMERGENCY: " + EMERGENCY_MESSAGE),
    "URGENT_24H": (st.warning, "URGENT: your pet should be seen by a vet within 24 hours."),
    "ROUTINE": (
        st.success,
        "ROUTINE: book a normal appointment. Contact a vet sooner if things get worse.",
    ),
}


@st.cache_resource
def setup_tracking() -> None:
    tracking.setup("pet-triage-ui")


def sample_photo() -> bytes:
    """A drawn placeholder, so the photo step can be demoed with one click."""
    img = Image.new("RGB", (480, 320), (222, 196, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse((170, 100, 310, 210), fill=(196, 72, 72), outline=(150, 40, 40), width=4)
    draw.text((16, 16), "SAMPLE PHOTO (placeholder)", fill=(40, 40, 40))
    draw.text((16, 290), "e.g. a red patch of skin", fill=(40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def reset() -> None:
    st.session_state.stage = "describe"  # describe -> questions -> (photo) -> done
    st.session_state.messages = []
    st.session_state.session = None
    st.session_state.run_id = None
    st.session_state.record = None
    st.session_state.photo = None
    st.session_state.error = None


# --- Talking to the triage core, with each conversation tracked as one MLflow run ---


def run_turn(fn, *args) -> TurnResult | None:
    ss = st.session_state
    try:
        with mlflow.start_run(run_id=ss.run_id, run_name=None if ss.run_id else "ui") as run:
            if ss.run_id is None:
                mlflow.log_params(tracking.config_params(ss.session.config))
            ss.run_id = run.info.run_id
            with st.spinner("Thinking... a local model can take 10-30 seconds."):
                result = fn(*args)
            tracking.log_turn(result.metrics)
        return result
    except Exception as err:  # show the problem rather than a stack trace
        with mlflow.start_run(run_id=ss.run_id):
            mlflow.log_metric("schema_valid", 0)
        ss.error = f"Something went wrong talking to the model: {err}"
        return None


def handle(result: TurnResult | None) -> None:
    ss = st.session_state
    if result is None:
        return
    if result.questions:
        text = "Thanks. A few quick questions so the vet has the full picture:\n\n"
        text += "\n".join(f"- {q}" for q in result.questions)
        ss.messages.append(("assistant", text))
        ss.stage = "questions"
        return

    ss.record = result.record
    if ss.record.urgency_tier != "EMERGENCY" and ss.record.requires_photo_upload:
        ss.messages.append(
            (
                "assistant",
                "Thank you. A photo would help the vet see this. Could you add one below?",
            )
        )
        ss.stage = "photo"
    else:
        finish()


def finish(photo: tuple[str, bytes] | None = None) -> None:
    ss = st.session_state
    ss.photo = photo
    if photo:
        ss.messages.append(("user", f"📷 Photo attached: {photo[0]}"))
    record = ss.record
    if record.skipped_questions_for_emergency:
        ss.messages.append(
            ("assistant", "I've skipped my questions because what you describe could be urgent.")
        )
    with mlflow.start_run(run_id=ss.run_id):
        mlflow.log_metric("schema_valid", 1)
        mlflow.log_metric("photo_attached", int(photo is not None))
        # Only photo metadata is logged: owner photos don't belong in experiment tracking.
        photo_info = {"attached": photo is not None, "name": photo[0] if photo else None}
        mlflow.log_dict(
            {
                "transcript": [m.model_dump() for m in ss.session.history],
                "photo": photo_info,
                **record.model_dump(),
            },
            "triage_record.json",
        )
    ss.stage = "done"


def start(owner_text: str, config: ModelConfig) -> None:
    ss = st.session_state
    ss.session = TriageSession(config)
    ss.messages.append(("user", owner_text))
    handle(run_turn(ss.session.start, owner_text))


def answer(owner_text: str) -> None:
    ss = st.session_state
    ss.messages.append(("user", owner_text))
    handle(run_turn(ss.session.answer, owner_text))


# --- Page ---

setup_tracking()
if "stage" not in st.session_state:
    reset()
ss = st.session_state

with st.sidebar:
    st.header("Settings")
    defaults = ModelConfig()
    model_name = st.text_input(f"Model ({defaults.provider})", defaults.model_name)
    versions = sorted(p.name for p in Path(PROMPTS_DIR).iterdir() if p.is_dir())
    prompt_version = st.selectbox(
        "Prompt version", versions, index=versions.index(defaults.prompt_version)
    )
    config = ModelConfig(model_name=model_name, prompt_version=prompt_version)
    if st.button("New conversation", width="stretch"):
        reset()
        st.rerun()

    st.header("Try an example")
    st.caption("Synthetic owner messages from the test cases.")
    for case in load_golden_cases():
        label = case["case_id"].split("_", 1)[1].replace("_", " ")
        if st.button(label, key=case["case_id"], width="stretch", disabled=ss.stage != "describe"):
            # Handled in the chat column so the "thinking" spinner shows there.
            ss.pending_example = case["owner_inputs"][0]

    st.divider()
    st.caption(
        "Runs are logged to MLflow, experiment **pet-triage-ui**. "
        "Open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`."
    )

st.title("🐾 Pet Triage Demo")
st.caption(
    "Gathers history and judges urgency before a vet consultation. "
    "It does not diagnose or suggest treatment. Prototype only, not clinically validated."
)

owner_col, vet_col = st.columns([3, 2], gap="large")

with owner_col:
    st.subheader("Owner's view")
    if not ss.messages:
        st.info("Describe what's going on with your pet, or pick an example on the left.")
    for role, text in ss.messages:
        with st.chat_message(role):
            st.markdown(text)
    if ss.error:
        st.error(ss.error)

    if ss.stage == "photo":
        with st.container(border=True):
            st.markdown(
                "**Add a photo** (demo: the photo is attached for the vet, nothing analyses it)"
            )
            uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
            c1, c2 = st.columns(2)
            if uploaded is not None:
                finish((uploaded.name, uploaded.getvalue()))
                st.rerun()
            if c1.button("Use a sample photo", width="stretch"):
                finish(("sample_photo.png", sample_photo()))
                st.rerun()
            if c2.button("Skip", width="stretch"):
                finish(None)
                st.rerun()

    if ss.stage == "describe":
        text = st.chat_input("Describe what's going on with your pet")
        text = text or ss.pop("pending_example", None)
        if text:
            start(text, config)
            st.rerun()
    elif ss.stage == "questions":
        text = st.chat_input("Your answers")
        if text:
            answer(text)
            st.rerun()
    elif ss.stage == "done":
        show, message = URGENCY_DISPLAY[ss.record.urgency_tier]
        show(message)
        st.button("Start a new conversation", on_click=reset)

with vet_col:
    st.subheader("Vet's view: triage record")
    record = ss.record
    if record is None:
        st.caption("The record appears here once the assistant has enough information.")
    else:
        show, _message = URGENCY_DISPLAY[record.urgency_tier]
        show(f"**{record.urgency_tier}**")
        st.markdown(f"#### {record.complaint_term} `{record.complaint_code}`")
        st.markdown(f"**Owner summary.** {record.owner_summary}")
        st.markdown(f"**Why this urgency.** {record.urgency_reason}")

        if record.model_urgency_tier != record.urgency_tier:
            st.warning(
                f"The model said {record.model_urgency_tier}; the keyword safety floor "
                f"raised it to {record.urgency_tier}."
            )
        if record.red_flags_detected:
            st.markdown("**Red flags found by the keyword check**")
            for check in record.red_flag_checks:
                agreement = record.flag_lift_agreement.get(check.flag_id)
                extra = f" (re-check agreement: {agreement:.0%})" if agreement is not None else ""
                quote = check.evidence_quote
                st.markdown(f'- `{check.flag_id}`: **{check.status}**{extra}, quote: "{quote}"')

        if ss.stage == "photo":
            st.info("Waiting for the owner's photo.")
        elif ss.photo:
            st.image(
                ss.photo[1], caption=f"Photo from owner: {ss.photo[0]} (not analysed)", width=240
            )
        elif record.requires_photo_upload:
            st.caption("Photo requested; the owner skipped it.")

        with st.expander("Full record (JSON)"):
            st.json(record.model_dump())
        if ss.run_id:
            st.caption(f"MLflow run: `{ss.run_id}`")
