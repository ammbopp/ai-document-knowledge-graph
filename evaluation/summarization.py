"""Summaries: text-only baseline vs the graph-augmented summary, scored on overlap and faithfulness."""
import re

from src.config import SUMMARY_TEXT_CHARS
from src.grounding import ungrounded_terms  # noqa: F401  (shared with the app; re-exported for tests)
from src.llm import generate
from src.pipeline import analyze_document

from .common import Report, as_bool, fmt_stat, get_embedder, get_nlp, judge_json, load_articles, md_table, ratio

NAME = "summarization"
DESCRIPTION = "Text-only vs graph-augmented summaries: ROUGE, claim-level faithfulness, ungrounded terms"

BASELINE_PROMPT = """You are an expert summarizer. Write a clear, executive-level summary of the text.
Do not invent or add outside knowledge; ground every fact in the text.
Format the output with clear bullet points or short paragraphs.

=== ORIGINAL TEXT ===
{text}

Summary:"""

CLAIM_PROMPT = """Check one claim from a summary against the article.

Article:
{text}

Claim: "{claim}"

Is the claim fully supported by the article (no contradiction, no added facts, no changed numbers or names)?
Answer with JSON: {{"supported": true}} or {{"supported": false}}"""

# ---------- pure helpers (unit-tested) ----------
def split_claims(summary, min_words=4):
    """One claim per bullet / sentence, with list markers and bold stripped."""
    claims = []
    for line in summary.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).replace("**", "").strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            if len(part.split()) >= min_words and not part.rstrip().endswith(":"):
                claims.append(part.strip())
    return claims


def faithfulness(summary, source):
    """-> (supported, judged). Claims the judge could not decide are left out."""
    supported = judged = 0
    for claim in split_claims(summary):
        verdict = as_bool((judge_json(CLAIM_PROMPT.format(text=source, claim=claim)) or {}).get("supported"))
        if verdict is not None:
            judged += 1
            supported += verdict
    return supported, judged


def run(args):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    nlp, embedder = get_nlp(), get_embedder()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)
    report = Report(NAME, "Summarization evaluation")

    for i, article in enumerate(articles, 1):
        print(f"[summarization {i}/{len(articles)}] {article.id}")
        result = analyze_document(article.text, nlp, embedder, lang="en")
        source = article.text[:SUMMARY_TEXT_CHARS]   # the same text both methods are allowed to see
        try:
            baseline = generate(BASELINE_PROMPT.format(text=source), temperature=0.1)
        except Exception as e:
            print(f"  baseline failed: {e}")
            continue
        for method, summary in (("text-only", baseline), ("graph-augmented", result.summary)):
            if not summary:
                continue
            supported, judged = faithfulness(summary, source)
            extra = ungrounded_terms(summary, source)
            row = {"article": article.id, "method": method, "summary_words": len(summary.split()),
                   "compression": ratio(len(summary.split()), article.words),
                   "claims_judged": judged, "claims_supported": supported,
                   "faithfulness": ratio(supported, judged), "ungrounded_terms": len(extra),
                   "ungrounded_examples": ", ".join(extra[:6])}
            if article.reference:
                for key, score in scorer.score(article.reference, summary).items():
                    row[f"{key}_f1"] = score.fmeasure
            report.rows.append(row)

    table = []
    for method in ("text-only", "graph-augmented"):
        rows = [r for r in report.rows if r["method"] == method]
        supported, judged = sum(r["claims_supported"] for r in rows), sum(r["claims_judged"] for r in rows)
        col = lambda key: [r.get(key) for r in rows]  # noqa: E731
        table.append([method, len(rows), fmt_stat(col("summary_words"), 0),
                      f"{ratio(supported, judged) * 100:.1f}% ({supported}/{judged})" if judged else "n/a",
                      fmt_stat(col("ungrounded_terms"), 1),
                      fmt_stat(col("rouge1_f1"), 3), fmt_stat(col("rouge2_f1"), 3), fmt_stat(col("rougeL_f1"), 3)])
    report.add_section("Summary quality", md_table(
        ["Method", "Articles", "Words", "Claims supported (pooled)", "Ungrounded terms / summary",
         "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"], table))
    report.notes += [
        "Faithfulness is claim-level: each bullet/sentence is judged separately against the article.",
        "'Ungrounded terms' are numbers and proper nouns in the summary that never appear in the article "
        "(deterministic, no LLM) -- a cheap hallucination signal that also catches invented dates.",
        "ROUGE is computed against the BBC description (one sentence), so absolute values are low; compare methods only.",
        "The graph-augmented summary comes from the full pipeline run (graph built from the same article).",
    ]
    return report
