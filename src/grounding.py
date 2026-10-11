"""Check a generated summary against its source: which names and numbers were not in the article?"""
import os
import re
from html import escape

from .relation_extraction_ai import _parse

_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_WORD = re.compile(r"[A-Za-z0-9]+")
_LIST_MARK = re.compile(r"^\s*(?:#+\s*|[-*•]\s+|\d+[.)]\s+)")      # a bullet needs a space: "**bold**" is not one
_WHOLE_BOLD = re.compile(r"^\*\*[^*]+\*\*:?$")                       # "**Key Points**" / "**Title: subtitle**"
_BOLD_LABEL = re.compile(r"^\*\*[^*]{1,60}:\*\*\s*|^\*\*[^*]{1,60}\*\*:\s*")   # "**Not for Beginners:** text"
_LABEL = re.compile(                                                 # "Note: ..." / "Key takeaway: ..." prefixes
    r"^(?:notes?|summary|key [a-z ]{0,20}|takeaways?|conclusion|overview|context|insights?|highlights?|disclaimer|"
    r"caveat|background|update|remarks?|important)[a-z ]{0,16}:\s+(?=\S)", re.IGNORECASE)
_BULLET_LABEL = re.compile(r"^[A-Za-z][A-Za-z' -]{0,28}:\s+(?=\S)")  # "Present: text" at the start of a bullet
_ENUMERATOR = re.compile(                                            # "Topic Shift 3", "2. Another point"
    r"\btopic shift\s+\d+|(?<![\w.,$£€])\d{1,2}[.)](?=\s+[A-Z\"“(])", re.IGNORECASE)

_OPENERS = {  # capitalised only because they start a sentence
    "overall", "however", "additionally", "furthermore", "moreover", "therefore", "finally", "meanwhile",
    "similarly", "specifically", "notably", "importantly", "ultimately", "currently", "today", "consequently",
    "thus", "hence", "instead", "still", "also", "indeed", "nevertheless", "nonetheless", "accordingly",
    "first", "second", "third", "lastly", "next", "then", "now", "here", "there", "despite", "according",
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}


def _without_headings(summary):
    """Drop section titles and strip list markers and "Note:"-style labels, keeping the sentences."""
    lines = []
    for line in summary.splitlines():
        is_bullet = bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s", line))
        text = _LIST_MARK.sub("", line).strip()
        if _WHOLE_BOLD.match(text):                       # a title on its own line
            continue
        text = _BOLD_LABEL.sub("", text).replace("**", "").strip()
        if not text or text.endswith(":") or (not is_bullet and len(text.split()) <= 4 and not text.endswith((".", "!", "?"))):
            continue
        text = _LABEL.sub("", text, count=1)
        if is_bullet:
            label = _BULLET_LABEL.match(text)
            if label and len(label.group().split()) <= 3:
                text = text[label.end():]
        lines.append(text)
    return "\n".join(lines)


def _number_key(token):
    return token.strip(".,").replace(",", "")


def _norm(word):
    """Lower-case and fold -ize/-ise spelling so Latinization matches Latinisation."""
    return word.lower().replace("z", "s")


def _related(a, b):
    """Same word allowing a short inflection or derivation (America/American, tariff/tariffs)."""
    prefix = len(os.path.commonprefix([a, b]))
    return prefix >= 5 and prefix >= max(len(a), len(b)) - 3


def ungrounded_terms(summary, source):
    """
    Numbers and proper nouns in the summary that never occur in the source, e.g. an invented date,
    amount or name. Section titles and labels are ignored; plurals, spelling variants and short
    inflections of source words, and numbers the source spells out ("nine"), count as grounded.
    It is still a heuristic: read it as "please verify", not as proof of error.
    """
    body = _ENUMERATOR.sub(" ", _without_headings(summary))
    if not body.strip() or not source:
        return []
    known = {_norm(w) for w in _WORD.findall(source)}
    by_start = {}
    for w in known:
        by_start.setdefault(w[:3], []).append(w)
    known_numbers = {_number_key(n) for n in _NUMBER.findall(source)}
    known_numbers |= {str(_NUMBER_WORDS[w]) for w in known if w in _NUMBER_WORDS}

    def grounded(term):
        term = _norm(term)
        return (term in known or term + "s" in known or (term.endswith("s") and term[:-1] in known)
                or any(_related(term, w) for w in by_start.get(term[:3], ())))

    missing = []
    for tok in _parse(body):
        if not tok.is_alpha or tok.is_stop:
            continue
        if tok.is_sent_start and tok.lower_ in _OPENERS:
            continue
        if (tok.pos_ == "PROPN" or (tok.is_title and not tok.is_sent_start)) and not grounded(tok.text):
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
