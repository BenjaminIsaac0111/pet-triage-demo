"""Pet triage prototype: the core logic, shared by the web demo, terminal chat and evaluation.

Where to start reading:

    conversation.py      the two-turn conversation: ask questions, then assess
    safety_floor.py      keyword check that can only raise urgency
    flag_recheck.py      re-asks the model before a red flag is dismissed
    schemas.py           the formats the model must fill in, and the final record

Supporting modules:

    chains.py            LangChain prompts and chains, with one retry on bad output
    chat_models.py       picks the model provider (Ollama, OpenAI or Anthropic)
    settings.py          every setting, read from environment variables
    reference_data.py    loads the JSON files in data/
    text_matching.py     shared text clean-up for phrase matching
    mlflow_tracking.py   what gets logged to MLflow
    scoring.py           scores a record against a test case
    outcome_matrix.py    expected against actual outcome, logged by the evaluation
    lexical_baseline.py  rules-only baseline, to show what the model adds
"""
