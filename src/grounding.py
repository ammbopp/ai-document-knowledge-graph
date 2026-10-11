"""Check a generated summary against its source: which names and numbers were not in the article?"""
import os
import re
from html import escape

from .relation_extraction_ai import _parse

_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_WORD = re.compile(r"[A-Za-z0-9]+")
_LIST_MARK = re.compile(r"^\s*(?:#+|[-*•]|\d+[.)])\s*")


def _without_headings(summary):
    """Drop section titles ("**Key Points:**", "## Context") and strip list markers."""
    lines = []
    for line in summary.splitlines():
        is_bullet = bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s", line))
        text = _LIST_MARK.sub("", line).replace("**", "").strip()
        if not text or text.endswith(":") or (not is_bullet and len(text.split()) <= 4 and not text.endswith((".", "!", "?"))):
            continue
        lines.append(text)
    return "\n".join(lines)


def _number_key(token):
    return token.strip(".,").replace(",", "")


def _related(a, b):
    """Same word allowing a short inflection or derivation (America/American, tariff/tariffs)."""
    prefix = len(os.path.commonprefix([a, b]))
    return prefix >= 5 and prefix >= max(len(a), len(b)) - 3


def ungrounded_terms(summary, source):
    """
    Numbers and proper nouns in the summary that never occur in the source, e.g. an invented date,
    amount or name. Section titles are ignored and short inflections of source words count as
    grounded, but it is still a heuristic: read it as "please verify", not as proof of error.
    """
    body = _without_headings(summary)
    if not body or not source:
        return []
    known = {w.lower() for w in _WORD.findall(source)}
    by_start = {}
    for w in known:
        by_start.setdefault(w[:3], []).append(w)
    known_numbers = {_number_key(n) for n in _NUMBER.findall(source)}

    def grounded(term):
        term = term.lower()
        return term in known or any(_related(term, w) for w in by_start.get(term[:3], ()))

    missing = []
    for tok in _parse(body):
        if tok.is_alpha and not tok.is_stop and (tok.pos_ == "PROPN" or (tok.is_title and not tok.is_sent_start)):
            if not grounded(tok.text):
                missing.append(tok.text)
    missing += [_number_key(n) for n in _NUMBER.findall(body) if _number_key(n) not in known_numbers]
    return sorted(set(missing))


def summary_flags(result):
    """Ungrounded terms of an AnalysisResult's summary (computed once, kept in result.stats)."""
    if "ungrounded" not in result.stats:
        result.stats["ungrounded"] = ungrounded_terms(result.summary or "", getattr(result, "text", ""))
    return result.stats["ungrounded"]


def highlight(html_text, terms, title=""):
    """Wrap each flagged term in <mark class="ungrounded"> (text only, tags are left untouched)."""
    if not terms:
        return html_text
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    # '&' and '#' in the lookbehind keep matches out of HTML entities such as &amp; or &#x27;
    pattern = re.compile(rf"(?<![\w.,&#])({alternatives})(?!\w)")
    tip = f' title="{escape(title, quote=True)}"' if title else ""
    parts = re.split(r"(<[^>]+>)", html_text)
    return "".join(p if p.startswith("<") else pattern.sub(rf'<mark class="ungrounded"{tip}>\1</mark>', p)
                   for p in parts)
