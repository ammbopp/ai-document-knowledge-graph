import re

from .config import COREF_CHUNK_CHARS
from .llm import generate
from .text_utils import split_by_chars

_OUTPUT = re.compile(r"<rewritten>(.*?)</rewritten>", re.DOTALL | re.IGNORECASE)
_LEAK_MARKERS = ("CRITICAL SAFETY RULES", "Original Text:", "Rewritten Text", "STRICT PLURAL", "<text>")

COREF_PROMPT = """You are a precise NLP Coreference Resolution engine.
Rewrite the text inside <text> tags, replacing pronouns (he, she, it, they, his, her, their) with their exact explicit noun antecedents.

Rules:
1. Plural pronouns ("they", "their") must be replaced by a plural group, never by a single individual's name.
2. Make sure each replacement is logical; do not assign every pronoun to the most frequent name.
3. Keep every other word and the sentence structure exactly unchanged.
4. Reply with ONLY the rewritten text wrapped in <rewritten></rewritten> tags. No rules, no commentary.

<text>
{text}
</text>
<rewritten>"""


def _resolve_chunk(chunk):
    try:
        out = generate(COREF_PROMPT.format(text=chunk), stop=["</rewritten>"])
    except Exception as e:
        print(f"⚠️ LLM Coreference Failed: {e}")
        return chunk
    match = _OUTPUT.search(out + "</rewritten>")
    out = (match.group(1) if match else out).strip()
    # Reject echoed prompts, and rewrites that summarised, truncated or padded the text
    if any(m in out for m in _LEAK_MARKERS) or not out:
        return chunk
    if not (0.7 * len(chunk) <= len(out) <= 1.4 * len(chunk)):
        return chunk
    return out


def resolve_coreferences_with_llm(text, max_chars=COREF_CHUNK_CHARS):
    """Resolve pronouns chunk by chunk so long documents are not truncated by the context window."""
    return "\n".join(_resolve_chunk(c) for c in split_by_chars(text, max_chars))
