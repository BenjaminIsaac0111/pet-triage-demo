# Pet Triage Demo

A pet owner describes a problem in their own words. The assistant asks 2–3 follow-up questions, then writes a short record for the vet: a complaint code, an urgency level (EMERGENCY, URGENT_24H or ROUTINE), and whether a photo would help. It never diagnoses or suggests treatment.

Built with LangChain, a local model through Ollama (or your own model or API key), and MLflow, which tracks every conversation and was used to compare prompt versions.

**Prototype only.** The test cases are mostly synthetic and labelled by a non-vet. This shows the approach, not clinical safety.

The service covers **cats and dogs only**, as an example scope. Anything else, or anything that isn't a health problem, is politely declined, except a likely emergency, which always gets "contact a vet now". Design decisions and their trade-offs are in [docs/adr](docs/adr/README.md).

```mermaid
graph LR
    A[Owner's message] --> B{Keyword check}
    B -- no emergency words --> C[LLM asks 2-3 questions]
    C --> D[Owner answers]
    D --> E[LLM fills triage record]
    B -- emergency words --> E
    E --> F{Safety floor}
    F --> G[Record for the vet]
    C -- not a cat or dog health problem --> X[Politely declined]
    F -- out of scope and not an emergency --> X
    E -. traces, timings, results .-> H[(MLflow)]
```

## How it works

1. **Keyword check.** The owner's words are scanned for emergency phrases ("nothing comes up", "struggling to breathe"). It's free and predictable, but it can't read context like "*not* struggling to breathe".
2. **Questions.** With no emergency phrase, the LLM asks 2–3 follow-up questions. With one, it skips straight to the assessment.
3. **Scope.** Before writing anything else, both LLM calls say which animal this is and whether it's a health problem. Code, not the model, applies the rule: anything that isn't a cat or dog health problem is politely declined, with no record and no appointment advice. A likely emergency is never declined. This was added after an exploratory test ("my racoon keeps stealing my food") was coded as "Lump or swelling".
4. **Assessment.** The LLM fills a fixed form: summary, complaint code, photo flag, urgency. Codes can only come from our list, enforced while the model writes; a problem that isn't on the list is coded "Other / not listed" rather than forced into the nearest wrong code. For each emergency phrase found, the model also says whether the sign is present, absent or unclear, quoting the owner. That gives the keyword check its context, in the same call.
5. **Safety floor.** Any emergency phrase forces EMERGENCY, unless the model says the sign is absent, its quote really appears in the owner's words, *and* it gives the same answer when asked 4 more times. The rule can only raise urgency, never lower it.
6. **Tracking.** Each conversation is one MLflow run: settings as parameters; latency, retries and safety checks as metrics; the record as a JSON file. LangChain autologging captures every prompt and response as a trace.

## Where the code lives

| File | What it does |
|---|---|
| `triage/conversation.py` | The two-turn conversation. **Start here.** |
| `triage/safety_floor.py` | The keyword check and the rule that can only raise urgency |
| `triage/flag_recheck.py` | Re-asks the model before an emergency flag is dismissed |
| `triage/schemas.py` | The forms the model must fill in, and the final record |
| `triage/chains.py` | LangChain prompts and chains, with one retry on bad output |
| `triage/mlflow_tracking.py` | What gets logged to MLflow |
| `evaluate.py` + `triage/scoring.py` | Runs the test cases and scores them |
| `web_demo.py`, `terminal_chat.py` | Thin front ends over the same conversation code |
| `prompts/v1` … `v3` | Prompt wording, kept out of the code so versions can be compared (v3, the default, adds the scope check) |
| `docs/adr/` | Design decisions: what was decided, the alternatives, and the trade-offs |
| `data/` | Complaint codes, emergency phrases, test cases and edge cases |

`triage/__init__.py` lists the smaller supporting modules. Planned improvements are marked `TODO` in the code, next to where they'd go.

## Results

Model `qwen2.5:7b-instruct` (Ollama weights `845dbda0ea48`), run locally at temperature 0, so anyone with Ollama can reproduce it. Six synthetic cases, each worded two ways (12 runs). Counts, not percentages, because the set is small: a difference of one or two runs is noise.

| | Rules only (no LLM) | Prompt v1 | Prompt v2 | Prompt v3 (default) |
|---|---|---|---|---|
| Emergencies caught | **3 / 4** | **3 / 4** | 4 / 4 | **3 / 4** |
| Rated less urgent than expected | **5 / 12** | 1 / 12 | 0 / 12 | 1 / 12 |
| Rated more urgent than expected | 1 / 12 | 1 / 12 | 1 / 12 | 2 / 12 |
| Complaint code correct | 7 / 12 | 11 / 12 | 11 / 12 | 10 / 12 |
| Photo flag correct | 11 / 12 | 10 / 12 | 10 / 12 | 10 / 12 |
| Real cases wrongly declined as out of scope | n/a | n/a | n/a | 0 / 12 |
| Edge cases handled (`--edge-cases`) | n/a | n/a | n/a | 7 / 7 |

- **Rules only** uses just the phrase lists. It couldn't grade the middle tier and flagged "*not* struggling to breathe" as an emergency. Most of what the LLM adds is coping with paraphrase and grading the middle tier.
- **The missed emergency is the same case almost every time:** a blocked cat described as "only a few drops of wee come out". No emergency phrase covers that wording, so it depends on the model alone, and only v2 caught it (once). It's a gap in the phrase list, not in any one prompt.
- **The over-triage is mostly deliberate.** "He's *not* struggling to breathe" is kept as an emergency when the re-check samples disagree: the safety floor prefers a false alarm to a missed emergency.
- **v1 → v2.** v2 added plain definitions of each urgency level and "when torn, choose the more urgent level", after v1 under-triaged a spreading skin sore in an earlier run on a larger model (`qwen3.8:27b`). v2 was written after seeing that failure, so its score is partly fitted to the test.
- **v2 → v3.** v3 adds the scope check (cats and dogs, health problems only) and the "Other / not listed" code. It handles all 7 edge cases, including declining the raccoon and still escalating a rabbit that can't breathe, without declining any real case. On the main cases it's one run behind v2 on most rows, which is within noise at this size; v2 is the only version that caught every emergency. Its one code miss on the edge cases: a dog's smelly ear was coded "Eye discharge" instead of "Other".
- **Versions are compared as written.** v1 and v2 run with the reply formats they were written for. Adding v3's scope fields to them made the 7B model decline real cases (vomiting, a hot spot) as "not a health problem", because their wording doesn't explain the fields: prompt and reply format are versioned together.
- **One real case.** My cat developed a painful lump after being neutered. I wrote the message before running anything and labelled it with what the vet nurse decided at the time (an appointment the next day). Both wordings matched with every prompt version, and no prompt was tuned on it. It also exposed a gap in the complaint list, so I added a "lump or swelling" code.

## Running it

Needs Python 3.11+ and one way to run a model.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install                 # secret scan and lint on commit, tests on push
ollama pull qwen2.5:7b             # skip if you use your own model or API
```

| | |
|---|---|
| Web demo | `streamlit run web_demo.py` |
| Terminal chat | `python terminal_chat.py` |
| Test cases, logged to MLflow | `python evaluate.py` (`--prompt-version v1` to compare, `--baseline lexical` for rules only, `--edge-cases` for scope checks) |
| Unit tests (no model needed) | `pytest` (`-m live` to also test against the real model) |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

**Choosing a model.** Copy `.env.example` to `.env` and set:

| To use | Set |
|---|---|
| Ollama (default) | nothing, or `TRIAGE_MODEL=<name from ollama list>` |
| A local OpenAI-compatible server (LM Studio, vLLM) | `TRIAGE_PROVIDER=openai`, `TRIAGE_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY=not-needed` |
| Anthropic | `TRIAGE_PROVIDER=anthropic`, `TRIAGE_MODEL`, `ANTHROPIC_API_KEY` |
| OpenAI | `TRIAGE_PROVIDER=openai`, `TRIAGE_MODEL`, `OPENAI_API_KEY` |

Keys stay in the git-ignored `.env` and are never logged. Results depend on the model, so rerun `python evaluate.py` on yours. With Docker (Ollama only): `docker compose up -d`, then `docker compose exec ollama ollama pull qwen2.5:7b`, and open http://localhost:8501. GitHub Actions runs the linter, the tests on Python 3.11 and 3.13, and a Docker build on every push.

## Limits and next steps

- **Not clinically validated.** It needs cases labelled by vets, then a period running silently alongside vets, before anyone relies on it.
- **Placeholder codes.** Real VeNom codes are licensed on request, so this uses made-up `DEMO-xxx` codes. Swapping them in is a data change.
- **Uncertainty.** Agreement between repeat answers shows how sure the model is, not whether it's right. With labelled data it could be calibrated (for example with conformal prediction) into a checked limit on missed emergencies.
- **Monitoring.** In production: watch for shifts in what owners write and what the system outputs, how often vets correct it, and results per species and language. Rerun `evaluate.py` before every prompt or model change.

## Glossary

- **Triage:** deciding how urgently a case needs attention. **Under-triage** (rating it less urgent than it is) is the dangerous mistake.
- **Red flag:** a sign that may mean an emergency, such as unproductive retching. **Lifting** a flag means dismissing it.
- **VeNom:** a standard UK list of veterinary terms and codes.
- **Structured output:** making the LLM reply in a fixed format with allowed values, instead of free text.
- **Temperature:** how much randomness the model uses when choosing words. At 0 it always picks the most likely word.
- **MLflow run / trace:** a run is one tracked experiment with its settings and results; a trace records each LLM call.

## License

[MIT](LICENSE)
