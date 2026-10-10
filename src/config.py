"""Central configuration (override with environment variables)."""
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))  # seconds per request
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

RESOLVER_THRESHOLD = 0.85
MIN_CONFIDENCE = 0.7
WINDOW_SIZE = 3
WINDOW_OVERLAP = 1
MIN_SENTENCE_LENGTH = 10

# Coreference / summarization budgets (characters)
COREF_CHUNK_CHARS = 1500
SUMMARY_TEXT_CHARS = 8000
SUMMARY_GRAPH_CHARS = 4000
