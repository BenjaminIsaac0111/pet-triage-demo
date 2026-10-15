"""Web demo: the owner's chat on the left, the vet's triage record on the right.

Usage: streamlit run web_demo.py

A thin layer over triage/conversation.py, the same code the terminal chat and
the evaluation use. Each conversation is one MLflow run in the "pet-triage-ui"
experiment.

Streamlit reruns this whole script on every click, so anything that must
survive between clicks is kept in `st.session_state` (see `reset_state`).
"""

import io
from pathlib import Path

import mlflow
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from triage import mlflow_tracking
from triage.chains import PROMPTS_DIR
from triage.conversation import EMERGENCY_MESSAGE, TriageConversation, TurnResult
from triage.reference_data import load_evaluation_cases
from triage.settings import TriageSettings

# How each urgency tier is shown: Streamlit's coloured box, and the advice for the owner.
URGENCY_DISPLAY = {
    "EMERGENCY": (st.error, "EMERGENCY: " + EMERGENCY_MESSAGE),
    "URGENT_24H": (st.warning, "URGENT: your pet should be seen by a vet within 24 hours."),
    "ROUTINE": (
        st.success,
        "ROUTINE: book a normal appointment. Contact a vet sooner if things get worse.",
    ),
}


# --- Conversation state, kept between Streamlit reruns ---


def reset_state() -> None:
    """Start a fresh conversation."""
    state = st.session_state
    state.stage = "describe"  # describe -> questions -> photo (only if requested) -> done
    state.messages = []  # (role, text) pairs shown in the chat
    state.conversation = None  # the TriageConversation in progress
    state.run_id = None  # the MLflow run this conversation logs to
    state.record = None  # the final TriageRecord, once there is one
    state.photo = None  # (file name, image bytes), if the owner added one
    state.error = None  # a message to show if the model call failed


def run_tracked_turn(take_turn, *turn_args) -> TurnResult | None:
    """Run one conversation turn inside this conversation's MLflow run.

    The same run is reopened on each turn, so one conversation is one run.
    Returns None, and stores a message to show, if the model call fails.
    """
    state = st.session_state
    try:
        with mlflow.start_run(run_id=state.run_id, run_name=None if state.run_id else "ui") as run:
            if state.run_id is None:
                mlflow.log_params(mlflow_tracking.run_params(state.conversation.settings))
            state.run_id = run.info.run_id
            with st.spinner("Thinking... a local model can take 10-30 seconds."):
                turn = take_turn(*turn_args)
            mlflow_tracking.log_turn_metrics(turn.metrics)
        return turn
    except Exception as error:  # show the problem rather than a stack trace
        with mlflow.start_run(run_id=state.run_id):
            mlflow.log_metric("schema_valid", 0)
        state.error = f"Something went wrong talking to the model: {error}"
        return None


def start_conversation(owner_message: str, settings: TriageSettings) -> None:
    state = st.session_state
    state.conversation = TriageConversation(settings)
    state.messages.append(("user", owner_message))
    show_turn(run_tracked_turn(state.conversation.start, owner_message))


def send_answers(owner_answers: str) -> None:
    state = st.session_state
    state.messages.append(("user", owner_answers))
    show_turn(run_tracked_turn(state.conversation.answer, owner_answers))


def show_turn(turn: TurnResult | None) -> None:
    """Move the conversation on: show the questions, ask for a photo, or finish."""
    state = st.session_state
    if turn is None:
        return
    if turn.questions:
        questions = "\n".join(f"- {question}" for question in turn.questions)
        state.messages.append(
            (
                "assistant",
                f"Thanks. A few quick questions so the vet has the full picture:\n\n{questions}",
            )
        )
        state.stage = "questions"
        return

    state.record = turn.record
    if state.record.urgency_tier != "EMERGENCY" and state.record.requires_photo_upload:
        state.messages.append(
            (
                "assistant",
                "Thank you. A photo would help the vet see this. Could you add one below?",
            )
        )
        state.stage = "photo"
    else:
        finish_conversation()


def finish_conversation(photo: tuple[str, bytes] | None = None) -> None:
    """Log the record to MLflow and show the outcome.

    Args:
        photo: (file name, image bytes) if the owner added one. Only whether a
            photo was attached is logged, never the image itself.
    """
    state = st.session_state
    state.photo = photo
    if photo:
        state.messages.append(("user", f"📷 Photo attached: {photo[0]}"))
    if state.record.skipped_questions_for_emergency:
        state.messages.append(
            ("assistant", "I've skipped my questions because what you describe could be urgent.")
        )
    with mlflow.start_run(run_id=state.run_id):
        mlflow.log_metrics({"schema_valid": 1, "photo_attached": int(photo is not None)})
        mlflow_tracking.log_conversation_record(
            state.conversation.history,
            state.record,
            extra={"photo": {"attached": photo is not None, "name": photo[0] if photo else None}},
        )
    state.stage = "done"


def drawn_sample_photo() -> bytes:
    """A drawn placeholder image, so the photo step can be demoed with one click."""
    image = Image.new("RGB", (480, 320), (222, 196, 160))
    draw = ImageDraw.Draw(image)
    draw.ellipse((170, 100, 310, 210), fill=(196, 72, 72), outline=(150, 40, 40), width=4)
    draw.text((16, 16), "SAMPLE PHOTO (placeholder)", fill=(40, 40, 40))
    draw.text((16, 290), "e.g. a red patch of skin", fill=(40, 40, 40))
    png = io.BytesIO()
    image.save(png, format="PNG")
    return png.getvalue()


# --- Page layout ---


def render_sidebar() -> TriageSettings:
    """Settings and example buttons. Returns the settings to use for a new conversation."""
    state = st.session_state
    with st.sidebar:
        st.header("Settings")
        defaults = TriageSettings()
        model_name = st.text_input(f"Model ({defaults.provider})", defaults.model_name)
        prompt_versions = sorted(p.name for p in Path(PROMPTS_DIR).iterdir() if p.is_dir())
        prompt_version = st.selectbox(
            "Prompt version", prompt_versions, index=prompt_versions.index(defaults.prompt_version)
        )
        if st.button("New conversation", width="stretch"):
            reset_state()
            st.rerun()

        st.header("Try an example")
        st.caption("Owner messages from the evaluation cases.")
        for case in load_evaluation_cases():
            label = case["case_id"].split("_", 1)[1].replace("_", " ")
            clicked = st.button(
                label, key=case["case_id"], width="stretch", disabled=state.stage != "describe"
            )
            if clicked:
                # Started from the chat column, so the "thinking" spinner shows there.
                state.pending_example = case["owner_inputs"][0]

        st.divider()
        st.caption(
            "Runs are logged to MLflow, experiment **pet-triage-ui**. "
            "Open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`."
        )
    return TriageSettings(model_name=model_name, prompt_version=prompt_version)


def render_photo_request() -> None:
    # TODO: Send photos to a vet (or an image model) for review, and store them
    # securely. Today the photo is only attached to the record.
    with st.container(border=True):
        st.markdown(
            "**Add a photo** (demo: the photo is attached for the vet, nothing analyses it)"
        )
        uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
        use_sample_column, skip_column = st.columns(2)
        if uploaded is not None:
            finish_conversation((uploaded.name, uploaded.getvalue()))
            st.rerun()
        if use_sample_column.button("Use a sample photo", width="stretch"):
            finish_conversation(("sample_photo.png", drawn_sample_photo()))
            st.rerun()
        if skip_column.button("Skip", width="stretch"):
            finish_conversation(None)
            st.rerun()


def render_owner_view(settings: TriageSettings) -> None:
    """The chat, plus whatever input the current stage needs."""
    state = st.session_state
    st.subheader("Owner's view")
    if not state.messages:
        st.info("Describe what's going on with your pet, or pick an example on the left.")
    for role, text in state.messages:
        with st.chat_message(role):
            st.markdown(text)
    if state.error:
        st.error(state.error)

    if state.stage == "photo":
        render_photo_request()
    elif state.stage == "describe":
        owner_message = st.chat_input("Describe what's going on with your pet")
        owner_message = owner_message or state.pop("pending_example", None)
        if owner_message:
            start_conversation(owner_message, settings)
            st.rerun()
    elif state.stage == "questions":
        owner_answers = st.chat_input("Your answers")
        if owner_answers:
            send_answers(owner_answers)
            st.rerun()
    elif state.stage == "done":
        show_box, advice = URGENCY_DISPLAY[state.record.urgency_tier]
        show_box(advice)
        st.button("Start a new conversation", on_click=reset_state)


def render_vet_view() -> None:
    """The triage record as the vet would see it."""
    state = st.session_state
    record = state.record
    st.subheader("Vet's view: triage record")
    if record is None:
        st.caption("The record appears here once the assistant has enough information.")
        return

    show_box, _advice = URGENCY_DISPLAY[record.urgency_tier]
    show_box(f"**{record.urgency_tier}**")
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
            agreement_note = (
                f" (re-check agreement: {agreement:.0%})" if agreement is not None else ""
            )
            st.markdown(
                f"- `{check.flag_id}`: **{check.status}**{agreement_note}, "
                f'quote: "{check.evidence_quote}"'
            )

    if state.stage == "photo":
        st.info("Waiting for the owner's photo.")
    elif state.photo:
        file_name, image_bytes = state.photo
        st.image(image_bytes, caption=f"Photo from owner: {file_name} (not analysed)", width=240)
    elif record.requires_photo_upload:
        st.caption("Photo requested; the owner skipped it.")

    with st.expander("Full record (JSON)"):
        st.json(record.model_dump())
    if state.run_id:
        st.caption(f"MLflow run: `{state.run_id}`")


@st.cache_resource
def start_tracking_once() -> None:
    """Streamlit reruns the script on every click; tracking only needs setting up once."""
    mlflow_tracking.start_tracking("pet-triage-ui")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Pet Triage Demo", page_icon="🐾", layout="wide")
    start_tracking_once()
    if "stage" not in st.session_state:
        reset_state()

    settings = render_sidebar()
    st.title("🐾 Pet Triage Demo")
    st.caption(
        "Gathers history and judges urgency before a vet consultation. "
        "It does not diagnose or suggest treatment. Prototype only, not clinically validated."
    )
    owner_column, vet_column = st.columns([3, 2], gap="large")
    with owner_column:
        render_owner_view(settings)
    with vet_column:
        render_vet_view()


main()
