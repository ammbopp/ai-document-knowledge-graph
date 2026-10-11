import requests

from .config import LLM_MODEL, LLM_NUM_CTX, LLM_TIMEOUT, OLLAMA_URL
from .i18n import t


def generate(prompt, temperature=0.0, *, json_mode=False, **options):
    """Call Ollama and return the response text (valid JSON when json_mode). Raises on HTTP/network errors."""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": LLM_NUM_CTX, **options},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def health(timeout=2.0, lang="th"):
    """Check that Ollama is reachable and the configured model is installed -> (ok, 'status — hint')."""
    try:
        response = requests.get(OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=timeout)
        response.raise_for_status()
    except Exception:
        return False, t("ollama_down", lang)
    names = [m.get("name", "") for m in response.json().get("models", [])]
    wanted = LLM_MODEL.split(":")[0]
    if not any(n == LLM_MODEL or n.split(":")[0] == wanted for n in names):
        return False, t("ollama_no_model", lang, model=LLM_MODEL)
    return True, t("ollama_ok", lang, model=LLM_MODEL)
