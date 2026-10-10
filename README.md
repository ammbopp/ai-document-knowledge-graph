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
3. Run the web app: `streamlit run app.py` (sidebar shows Ollama status and advanced settings;
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
`evaluate_*.py` and `run_10_experiments.py` at the repo root benchmark individual stages
(they need `datasets`, `rouge-score` and a running Ollama).
