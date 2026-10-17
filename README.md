# Pet Triage Demo

A pet owner picks cat, dog or another animal, then describes a problem in their own words. The assistant asks 2–3 follow-up questions, then writes a short record for the vet: a complaint code, an urgency level (EMERGENCY, URGENT_24H or ROUTINE), and whether a photo would help. It never diagnoses or suggests treatment.

Built with LangChain, a local model through Ollama (or your own model or API key), and MLflow, which tracks every conversation and was used to compare prompt versions.

**The design rule: the LLM proposes, code decides.** The LLM does what only it can do: read messy owner text and ask good questions. Every decision that ends a conversation or changes urgency is made by plain code, from what the owner chose or from the model's structured answers, so it's predictable and traceable.

**Prototype only.** The test cases are mostly synthetic and labelled by a non-vet. This shows the approach, not clinical safety.

The service covers **cats and dogs only**, as an example scope. The owner picks the animal, so scope never depends on the model: another animal is politely declined without calling it, unless an emergency phrase is found, which gets "contact a vet now". A cat or dog is never turned away.

```mermaid
graph LR
    S[Owner picks cat, dog or other] --> A[Owner's message]
    A --> B{Keyword check}
    B -- other animal --> X[Declined, or 'contact a vet now' if emergency words]
    B -- cat or dog, no emergency words --> C[LLM asks 2-3 questions]
    C --> D[Owner answers]
    D --> E[LLM fills triage record]
    B -- cat or dog, emergency words --> E
    E --> F{Safety floor and photo rule}
    F --> G[Record for the vet]
    E -. traces, timings, results .-> H[(MLflow)]
```

## How it works

1. **The owner picks the animal.** Cat, dog or another animal. Another animal is declined by code, with no model call; the keyword check still runs, so "my rabbit is struggling to breathe" gets "contact a vet now". For a cat or dog, the model is told the animal, so it doesn't ask.
2. **Keyword check.** The owner's words are scanned for emergency phrases ("nothing comes up", "struggling to breathe"). It's free and predictable, but it can't read context like "*not* struggling to breathe".
3. **Questions.** With no emergency phrase, the LLM asks 2–3 follow-up questions. With one, it skips straight to the assessment.
4. **Assessment.** The LLM fills a fixed form: summary, complaint code, urgency. Codes can only come from our list, enforced while the model writes; a problem that isn't on the list is coded "Other / not listed" rather than forced into the nearest wrong code. For each emergency phrase found, the model also says whether the sign is present, absent or unclear, quoting the owner. That gives the keyword check its context, in the same call.
5. **Safety floor.** Any emergency phrase forces EMERGENCY, unless the model says the sign is absent, its quote really appears in the owner's words (with any "not" next to the same word the owner put it next to), *and* it gives the same answer when asked 4 more times. The rule can only raise urgency, never lower it.
6. **Photo request.** Code decides: a photo is requested for skin, wound and eye complaints, never in an emergency. On the earlier runs, this rule matched the expected answer more often than the model's own choice did.
7. **Tracking.** Each conversation is one MLflow run: settings as parameters; latency, retries and safety checks as metrics; the record as a JSON file. LangChain autologging captures every prompt and response as a trace.

## Where the code lives

| File | What it does |
|---|---|
| `triage/conversation.py` | The two-turn conversation. **Start here.** |
| `triage/safety_floor.py` | The keyword check and the rule that can only raise urgency |
| `triage/flag_recheck.py` | Re-asks the model before an emergency flag is dismissed |
| `triage/schemas.py` | The forms the model must fill in, the final record, and the photo rule |
| `triage/chains.py` | LangChain prompts and chains, with one retry on bad output, and which fields each prompt version uses |
| `triage/mlflow_tracking.py` | What gets logged to MLflow |
| `evaluate.py` + `triage/scoring.py` | Runs the test cases and scores them |
| `web_demo.py`, `terminal_chat.py` | Thin front ends over the same conversation code |
| `prompts/v1` … `v4` | Prompt wording, kept out of the code so versions can be compared (v4 is the default) |
| `data/` | Complaint codes, emergency phrases, test cases and edge cases |

`triage/__init__.py` lists the smaller supporting modules. Planned improvements are marked `TODO` in the code, next to where they'd go.

## Results

Model `qwen2.5:7b` (Ollama weights `845dbda0ea48`, 7.6B, Q4_K_M), run locally at temperature 0 on an RTX 3090, about 1.5 s per conversation. Six synthetic cases and one real case, each worded two ways (14 runs): exactly what `python evaluate.py` runs. Counts, not percentages, because the set is small: a difference of one or two runs is noise. In MLflow these are the parent runs tagged `study = "readme-results"` (v4 ran a day after the others, with the same model, settings and cases).

| | Rules only (no LLM) | Prompt v1 | Prompt v2 | Prompt v3 | Prompt v4 (default) |
|---|---|---|---|---|---|
| Emergencies caught | 3 / 4 | 3 / 4 | 3 / 4 | 3 / 4 | 3 / 4 |
| … by the model alone, before the safety floor | n/a | 1 / 4 | 2 / 4 | 3 / 4 | 2 / 4 |
| Rated less urgent than expected | **7 / 14** | 1 / 14 | 1 / 14 | 1 / 14 | 1 / 14 |
| Rated more urgent than expected | 1 / 14 | 1 / 14 | 1 / 14 | 2 / 14 | 1 / 14 |
| Complaint code correct | 9 / 14 | 12 / 14 | 12 / 14 | 13 / 14 | 12 / 14 |
| Photo flag correct | 13 / 14 | 12 / 14 | 12 / 14 | 12 / 14 | 13 / 14 |
| Real cases wrongly declined as out of scope | 0 / 14 (can't decline) | 0 / 14 (can't decline) | 0 / 14 (can't decline) | 0 / 14 | 0 / 14 (never declines a cat or dog) |
| Out-of-scope requests declined (edge cases) | 0 / 3 | 0 / 3 | 0 / 3 | 3 / 3 | 2 / 3 |
| Edge cases ending as expected (`--edge-cases`) | 5 / 8 | 5 / 8 | 5 / 8 | 7 / 8 | 7 / 8 |
| Input tokens per conversation (mean) | none | 798 | 929 | 1,131 | 904 |

- **Rules only** uses just the phrase lists. It couldn't grade the middle tier (half its runs were rated less urgent than expected) and flagged "*not* struggling to breathe" as an emergency. Most of what the LLM adds is coping with paraphrase and grading the middle tier.
- **The missed emergency is the same case in every version:** a blocked cat described as "only a few drops of wee come out". No emergency phrase covers that wording, so it depends on the model alone, and no version caught it in these runs (v2 caught it once in an earlier one). It's a gap in the phrase list, not in any one prompt.
- **The safety floor earns its place.** With v1 the model alone rated only 1 of the 4 emergency runs as EMERGENCY; the keyword floor raised two more. The 7B model often leaves the red-flag checklist empty, and a flag the model doesn't answer about is kept.
- **The over-triage is mostly deliberate.** "He's *not* struggling to breathe" ends as EMERGENCY in every version, because the model skipped the flag check and the floor keeps an unanswered flag: it prefers a false alarm to a missed emergency. v3's second over-triage is a morning-bile case rated URGENT_24H.
- **v1 → v2.** v2 added plain definitions of each urgency level and "when torn, choose the more urgent level", after v1 under-triaged a spreading skin sore in an earlier run on a larger model (`qwen3.8:27b`). v2 was written after seeing that failure, so its score is partly fitted to the test. On these runs v1 and v2 tie on every final-result row; the difference shows only in what the model catches before the floor.
- **v2 → v3.** v3 had the model decide scope (which animal, and whether it's a health problem) and added the "Other / not listed" code. Without it, every earlier version triages whatever it's given: v2 still codes the raccoon as "Lump or swelling", exactly as in the original exploratory test. v3 declines the raccoon, a hamster and a lead-training question, and still escalates a rabbit that can't breathe. But the model's judgment wobbles: it declined a dog that suddenly started eating stones as "not a health problem" in one of three runs.
- **v3 → v4: the owner decides scope, code decides the photo.** The owner picks the animal, so another animal is declined with no model call (the raccoon, hamster and rabbit each took about 0.1 s) and a cat or dog can never be refused. The model no longer writes the species, health-concern or photo fields, so the prompt is shorter: about 20% fewer input tokens, and a median of 1.46 s per conversation against 1.71 s. The photo rule got 13/14 against the model's 12/14; its one miss is a cut paw coded as lameness, so the rule inherits the model's code errors. Code accuracy and over-triage moved by one run each, which is noise. **The trade:** the lead-training question now reaches the vet as "Other" instead of being declined. That's v4's one edge-case miss, and it's the cautious kind: a vet's minute, rather than a sick animal turned away.
- **A pet's name isn't a red flag.** A manual test found that "my cat Lily has been sneezing" was escalated to EMERGENCY, because "lily" was a poisoning phrase on its own. It now needs context ("a lily", "lily pollen"), and edge case EC08 checks it. The same review tightened the quote check so a "not" must sit where the owner put it. Neither change moved the numbers above: the model filled in only 4 flag checks across these runs, all "present".
- **One edge case no version gets right:** a dog's smelly ear gets a wrong code ("Eye discharge") instead of "Other", so the pull towards the nearest wrong code remains.
- **Versions are compared as written.** Each prompt version runs with the reply format it was written for: adding v3's scope fields to v1 and v2 made the 7B model decline real cases (vomiting, a hot spot) as "not a health problem", because their wording doesn't explain the fields. Every version is given the owner's pick, but only v4 uses it to decide scope.
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
| Test cases, logged to MLflow | `python evaluate.py` (`--prompt-version v3` to compare, `--baseline lexical` for rules only, `--edge-cases` for scope checks) |
| Compare local models (same cases, one MLflow study tag) | `python run_model_comparison.py --models qwen2.5:7b qwen3:8b` |
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
- **Grounding the questions.** The follow-up questions are the only model text the owner sees. Next: have the model choose from fixed question topics (onset, eating, breathing, urination…), and check the wording in code, with template questions as a fallback.
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
