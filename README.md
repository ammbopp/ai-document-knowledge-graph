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
   results can be exported as one full HTML report (summary, interactive graph, insights, relations table),
   or separately as HTML graph, CSV/JSON relations and a Markdown summary)
   or the CLI: `python -m src.main data/sample_04.txt` (writes `output/knowledge_graph.html`)
4. Tests: `pytest`

Settings can be overridden with env vars: `OLLAMA_URL`, `LLM_MODEL`, `LLM_TIMEOUT`, `LLM_NUM_CTX`
(see `src/config.py`).

## Documentation
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (Thai) explains how the whole system works, with five diagrams
(pipeline, extraction gates, entity resolution, module map, run sequence) as Mermaid and as SVG/PNG in
[docs/diagrams/](docs/diagrams/). Benchmark results: [evaluation/REPORT.md](evaluation/REPORT.md).

## Project layout
```
app.py              Streamlit entry point (wires the UI parts together)
ui/                 Web UI by screen area: sidebar, composer (input), analysis (run + progress), results, common
src/                Pipeline: preprocessing, relation extraction, entity resolution, graph, insights, summary,
                    plus graph_view / graph_layout / report (HTML output) and i18n / themes
assets/style.css    Web app styles (colour tokens come from src/themes.py)
assets/graph/       Interactive graph viewer: viewer.css and viewer.js (injected into the PyVis page)
docs/               Architecture write-up (ARCHITECTURE.md) and diagrams (diagrams/*.mmd, .svg, .png)
evaluation/         Benchmarks: python -m evaluation <experiment>, REPORT.md, gold/ (hand-labelled pairs)
tests/              pytest suite (no Ollama needed)
data/               Sample articles
```

## Limitations
- English only (the UI is Thai, but extraction models are English).
- Output quality depends on the LLM; the verifier reduces but does not eliminate wrong triples.
- The summary is written by the LLM and can add facts. Numbers and proper nouns that never appear in the
  source are highlighted (and listed in a warning) in the app and in the exported report; this word-level
  check (`src/grounding.py`) catches invented dates, amounts and names, not wrong wording.
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
| `--gold FILE`, `--llm-judge` | `resolution` only: hand-labelled pairs (default `evaluation/gold/resolution_pairs.json`); let the LLM judge unlabelled pairs (lenient) |

Results of the last full run, with method notes and caveats: [evaluation/REPORT.md](evaluation/REPORT.md).

Each run prints a Markdown report and saves it with the per-article rows as CSV. Articles are taken in a
fixed order and the LLM runs at temperature 0, so runs are comparable. Judge calls that fail are excluded
instead of being scored, and the report states how many were. The extractor and the judge are the same
model, so read absolute precision/faithfulness as optimistic and compare configurations with each other.
