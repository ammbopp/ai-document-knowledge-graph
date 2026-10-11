# AI Document Knowledge Graph

Extracts open-ended relationships from English documents with a local LLM (Ollama + Llama 3) and
builds an interactive knowledge graph, graph insights and a graph-augmented summary.

## Pipeline
Text → LLM coreference resolution (chunked) → sentence split + sliding windows →
LLM relation extraction (structural filters, passive→active normalisation, same-sentence grounding,
LLM verifier + two-order subject/object direction check, NER-based entity types, dates as Time nodes) →
entity resolution (per document) → MultiDiGraph → insights → PyVis visualization → summary

## Setup
1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull llama3`
2. `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
3. Run the web app: `streamlit run app.py` (sidebar: Thai/English switch, Ollama status, colour theme picker, advanced settings;
   results can be exported as HTML graph, CSV/JSON relations and a Markdown summary)
   or the CLI: `python -m src.main data/sample_04.txt` (writes `output/knowledge_graph.html`)
4. Tests: `pytest`

Settings can be overridden with env vars: `OLLAMA_URL`, `LLM_MODEL`, `LLM_TIMEOUT`, `LLM_NUM_CTX`
(see `src/config.py`).

## Limitations
- English only (the UI is Thai, but extraction models are English).
- Output quality depends on the LLM; the verifier reduces but does not eliminate wrong triples.
- Several LLM calls per chunk and per triple make long documents slow.

## Evaluation
Benchmarks live in the `evaluation/` package and run the production code (`src/`) directly, so they
always measure the current system. They need `datasets`, `rouge-score` and a running Ollama.

```
python -m evaluation --help
python -m evaluation system -n 5                  # whole pipeline: latency per stage, yield, graph, ROUGE
python -m evaluation preprocessing -n 3           # chunking / coreference ablation, core-fact recall
python -m evaluation extraction -n 3 --windows 6  # raw LLM vs structural filters vs verifier: precision
python -m evaluation resolution -n 3              # entity clustering P/R/F1 per threshold
python -m evaluation summarization -n 3           # text-only vs graph-augmented: faithfulness, ROUGE
python -m evaluation security                     # prompt-injection leakage / hijack rates
python -m evaluation all --source local           # everything on data/*.txt, no dataset download
```

| Option | Meaning |
| --- | --- |
| `-n, --samples` | articles per experiment (security: attacks per module) |
| `--source bbc\|local` | BBC News dataset (default) or `data/*.txt` (no reference summaries) |
| `--month`, `--offset` | pick another slice of the dataset |
| `--out DIR` | where to write results (default `evaluation/results/`, git-ignored) |

Each run prints a Markdown report and saves it with the per-article rows as CSV. Articles are taken in a
fixed order and the LLM runs at temperature 0, so runs are comparable. Judge calls that fail are excluded
instead of being scored, and the report states how many were. The extractor and the judge are the same
model, so read absolute precision/faithfulness as optimistic and compare configurations with each other.
