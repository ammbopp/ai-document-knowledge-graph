"""Pre-processing ablation: fixed chunks vs sliding windows vs coreference + sliding windows."""
import difflib
import re

from src.config import WINDOW_OVERLAP, WINDOW_SIZE
from src.llm import generate
from src.pipeline import extract_chunks
from src.preprocessing import resolve_coreferences_with_llm
from src.relation_extraction_ai import parse_triples
from src.text_utils import create_sliding_windows, split_sentences

from .common import Report, as_bool, fmt_stat, get_nlp, judge_json, load_articles, md_table, ratio

NAME = "preprocessing"
DESCRIPTION = "Chunking / coreference ablation: recall of core facts and pronoun replacement accuracy"

PRONOUNS = {"he", "she", "it", "they", "his", "her", "hers", "its", "their", "theirs", "him", "them",
            "himself", "herself", "itself", "themselves"}
MAX_PRONOUN_JUDGEMENTS = 12   # per article
MAX_EXTRACTED_LINES = 200     # keep the coverage prompt inside the context window

GOLD_PROMPT = """Read the text and extract ONLY the 5-7 essential factual triplets that capture its main points and central storyline.
Ignore minor background details and side mentions. Each fact must be explicit in the text.

Format, one per line:
Subject | Relation | Object

Text:
{text}

Triplets:"""

COVERAGE_PROMPT = """Decide whether a reference fact is covered by a list of extracted triplets.

Reference fact: {gold}

Extracted triplets:
{extracted}

The fact is covered if at least one extracted triplet (or a combination of two) states the same thing,
even with different wording. Answer with JSON: {{"covered": true}} or {{"covered": false}}"""

PRONOUN_PROMPT = """A coreference system replaced a pronoun with a noun phrase.

Passage before: "{before}"
Pronoun: "{pronoun}"
Replacement: "{replacement}"

Does the replacement name the same entity that the pronoun refers to in the passage?
Answer with JSON: {{"correct": true}} or {{"correct": false}}"""


# ---------- pure helpers (unit-tested) ----------
def standard_chunks(text, words_per_chunk=50):
    tokens = text.split()
    return [" ".join(tokens[i:i + words_per_chunk]) for i in range(0, len(tokens), words_per_chunk)]


def tokenize(text):
    """[(token, start, end)] words and punctuation with their character offsets."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def pronoun_replacements(original, rewritten):
    """
    Pronouns the rewrite changed: [{pronoun, replacement, before}] where `before` is the original
    passage around the pronoun. Found by token-level diff, so untouched text never counts.
    """
    a, b = tokenize(original), tokenize(rewritten)
    matcher = difflib.SequenceMatcher(None, [t[0].lower() for t in a], [t[0].lower() for t in b], autojunk=False)
    found = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or not all(t[0].lower() in PRONOUNS for t in a[i1:i2]) or j1 == j2:
            continue
        start, end = a[i1][1], a[i2 - 1][2]
        found.append({
            "pronoun": original[start:end],
            "replacement": rewritten[b[j1][1]:b[j2 - 1][2]],
            "before": original[max(0, start - 250):end + 100].strip(),
        })
    return found


def count_pronouns(text):
    return sum(1 for token, _, _ in tokenize(text) if token.lower() in PRONOUNS)


def preservation(original, rewritten):
    """Share of tokens kept in order (1.0 = identical text); guards against rewrites that drift."""
    a = [t[0].lower() for t in tokenize(original)]
    b = [t[0].lower() for t in tokenize(rewritten)]
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def format_triples(relations):
    return [f"{r['head']} | {r['relation']} | {r['tail']}" for r in relations]


# ---------- LLM-backed steps ----------
def gold_triples(text):
    try:
        out = generate(GOLD_PROMPT.format(text=text), temperature=0.0, num_predict=400)
    except Exception:
        return []
    return [" | ".join(t) for t in parse_triples(out.replace("\n", " ### "))]


def coverage(gold, extracted_lines):
    """-> (covered, total, judge_failures). Each gold fact is judged separately."""
    listing = "\n".join(extracted_lines[:MAX_EXTRACTED_LINES]) or "(none)"
    covered = failures = 0
    for fact in gold:
        verdict = as_bool((judge_json(COVERAGE_PROMPT.format(gold=fact, extracted=listing)) or {}).get("covered"))
        if verdict is None:
            failures += 1
        else:
            covered += verdict
    return covered, len(gold) - failures, failures


def judge_pronouns(replacements):
    """-> (correct, judged). Unjudgeable replacements are skipped, not counted as right or wrong."""
    correct = judged = 0
    for item in replacements[:MAX_PRONOUN_JUDGEMENTS]:
        verdict = as_bool((judge_json(PRONOUN_PROMPT.format(
            before=item["before"], pronoun=item["pronoun"], replacement=item["replacement"])) or {}).get("correct"))
        if verdict is not None:
            judged += 1
            correct += verdict
    return correct, judged


def run(args):
    nlp = get_nlp()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)
    groups = ["standard", "sliding", "coref+sliding"]
    report = Report(NAME, "Pre-processing ablation")
    per_group = {g: [] for g in groups}
    pronoun_rows = []

    for i, article in enumerate(articles, 1):
        print(f"[preprocessing {i}/{len(articles)}] {article.id}")
        gold = gold_triples(article.text)
        if not gold:
            print("  no gold triples, skipping")
            continue

        rewritten = resolve_coreferences_with_llm(article.text)
        inputs = {
            "standard": standard_chunks(article.text),
            "sliding": create_sliding_windows(split_sentences(nlp, article.text), WINDOW_SIZE, WINDOW_OVERLAP),
            "coref+sliding": create_sliding_windows(split_sentences(nlp, rewritten), WINDOW_SIZE, WINDOW_OVERLAP),
        }
        for group, chunks in inputs.items():
            relations = [r for rs in extract_chunks(chunks) for r in rs]
            matched, total, failures = coverage(gold, format_triples(relations))
            row = {"article": article.id, "group": group, "chunks": len(chunks), "triples": len(relations),
                   "gold": len(gold), "judged": total, "matched": matched, "judge_failures": failures,
                   "recall": ratio(matched, total)}
            per_group[group].append(row)
            report.rows.append(row)

        replacements = pronoun_replacements(article.text, rewritten)
        correct, judged = judge_pronouns(replacements)
        n_pron = count_pronouns(article.text)
        pronoun_rows.append({
            "article": article.id, "pronouns": n_pron, "replaced": len(replacements),
            "replace_rate": ratio(len(replacements), n_pron), "judged": judged, "correct": correct,
            "accuracy": ratio(correct, judged), "preservation": preservation(article.text, rewritten),
        })

    table = []
    for group in groups:
        rows = per_group[group]
        matched, judged = sum(r["matched"] for r in rows), sum(r["judged"] for r in rows)
        table.append([group, fmt_stat([r["triples"] for r in rows], 1), fmt_stat([r["recall"] for r in rows], 1, pct=True),
                      f"{ratio(matched, judged) * 100:.1f}% ({matched}/{judged})" if judged else "n/a",
                      f"{(judged - matched) / len(rows):.1f}" if rows else "n/a"])
    report.add_section("Core-fact recall by input preparation", md_table(
        ["Input", "Triples / article", "Recall (per article)", "Recall (pooled)", "Missed gold facts / article"], table))

    correct, judged = sum(r["correct"] for r in pronoun_rows), sum(r["judged"] for r in pronoun_rows)
    report.add_section("Coreference module", md_table(
        ["Metric", "Value"],
        [["Pronouns in original text", fmt_stat([r["pronouns"] for r in pronoun_rows], 1)],
         ["Pronouns replaced (share)", fmt_stat([r["replace_rate"] for r in pronoun_rows], 1, pct=True)],
         ["Replacement accuracy (judged, pooled)", f"{ratio(correct, judged) * 100:.1f}% ({correct}/{judged})" if judged else "n/a"],
         ["Text preserved outside replacements", fmt_stat([r["preservation"] for r in pronoun_rows], 1, pct=True)]]))
    failures = sum(r["judge_failures"] for rs in per_group.values() for r in rs)
    report.notes += [
        f"{len(pronoun_rows)} articles; gold = 5-7 core triples written by the same LLM from the full article, "
        "each judged separately against every group's extracted triples.",
        f"Judge calls that failed were excluded, not scored ({failures} coverage failures).",
        "The same model extracts, writes the gold set and judges, so absolute numbers are optimistic; "
        "compare groups with each other.",
    ]
    return report
