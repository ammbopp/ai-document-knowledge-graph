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


def health(timeout=2.0):
    """Check that Ollama is reachable and the configured model is installed -> (ok, message)."""
    try:
        response = requests.get(OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=timeout)
        response.raise_for_status()
    except Exception:
        return False, "เชื่อมต่อ Ollama ไม่ได้ — เปิดด้วยคำสั่ง `ollama serve`"
    names = [m.get("name", "") for m in response.json().get("models", [])]
    wanted = LLM_MODEL.split(":")[0]
    if not any(n == LLM_MODEL or n.split(":")[0] == wanted for n in names):
        return False, f"ยังไม่มีโมเดล `{LLM_MODEL}` — ติดตั้งด้วย `ollama pull {LLM_MODEL}`"
    return True, f"Ollama พร้อมใช้งาน ({LLM_MODEL})"
