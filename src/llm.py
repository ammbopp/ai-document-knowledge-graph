import requests

from .config import LLM_MODEL, LLM_NUM_CTX, LLM_TIMEOUT, OLLAMA_URL


def generate(prompt, temperature=0.0, **options):
    """Call Ollama and return the response text. Raises on HTTP/network errors."""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": LLM_NUM_CTX, **options},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
    response.raise_for_status()
    return response.json().get("response", "").strip()
