"""End-to-end run of the production pipeline: speed, extraction yield, graph shape and summary quality."""
import math

from src.pipeline import analyze_document

from .common import Report, fmt_stat, get_embedder, get_nlp, load_articles, md_table, ratio

NAME = "system"
DESCRIPTION = "Whole pipeline: latency per stage, triple yield, graph metrics, ROUGE vs reference"


def run(args):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    nlp, embedder = get_nlp(), get_embedder()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)

    report = Report(NAME, "System evaluation (production pipeline)")
    for i, article in enumerate(articles, 1):
        print(f"[system {i}/{len(articles)}] {article.id} ({article.words} words)")
        result = analyze_document(article.text, nlp, embedder, max_chars=args.max_chars or None, lang="en")
        s, ins = result.stats, result.insights

        row = {
            "article": article.id,
            "words": article.words,
            "sentences": s["sentences"],
            "chunks": s["chunks"],
            "total_sec": round(s["elapsed"], 2),
            "words_per_sec": round(article.words / s["elapsed"], 2) if s["elapsed"] else math.nan,
            **{f"sec_{stage}": round(sec, 2) for stage, sec in s["stage_seconds"].items()},
            "triples_extracted": s["extracted"],
            "triples_usable": s["usable"],
            "triples_in_graph": len(result.relations),
            "pass_rate": ratio(s["usable"], s["extracted"]),
            "triples_per_100_words": 100 * ratio(len(result.relations), article.words),
            "nodes": ins.get("total_entities", 0),
            "edges": ins.get("total_relations", 0),
            "density": ins.get("density", math.nan),
            "components": ins.get("num_connected_components", 0),
            "communities": ins.get("num_communities", 0),
            "summary_words": len(result.summary.split()),
        }
        if article.reference and result.summary:
            for key, score in scorer.score(article.reference, result.summary).items():
                row[f"{key}_f1"] = score.fmeasure
        report.rows.append(row)

    rows = report.rows
    if not rows:
        raise SystemExit("No articles were loaded.")
    col = lambda key: [r.get(key) for r in rows]  # noqa: E731
    stages = [k for k in rows[0] if k.startswith("sec_")] if rows else []

    report.add_section("Speed", md_table(
        ["Metric", "Mean ± std"],
        [["Total seconds / article", fmt_stat(col("total_sec"))],
         ["Words / second", fmt_stat(col("words_per_sec"))]]
        + [[f"Stage: {k[4:]} (s)", fmt_stat(col(k))] for k in stages]))
    report.add_section("Extraction yield", md_table(
        ["Metric", "Mean ± std"],
        [["Triples after structural + verifier gates", fmt_stat(col("triples_extracted"), 1)],
         ["Triples with confidence >= usable threshold",fmt_stat(col("triples_usable"), 1)],
         ["Triples in final graph", fmt_stat(col("triples_in_graph"), 1)],
         ["Pass rate (usable / extracted)", fmt_stat(col("pass_rate"), 1, pct=True)],
         ["Triples per 100 words", fmt_stat(col("triples_per_100_words"), 1)]]))
    report.add_section("Graph", md_table(
        ["Metric", "Mean ± std"],
        [["Nodes", fmt_stat(col("nodes"), 1)], ["Edges", fmt_stat(col("edges"), 1)],
         ["Density", fmt_stat(col("density"), 4)],
         ["Connected components", fmt_stat(col("components"), 1)],
         ["Communities", fmt_stat(col("communities"), 1)]]))
    if any("rougeL_f1" in r for r in rows):
        report.add_section("Summary vs reference (ROUGE F1)", md_table(
            ["Metric", "Mean ± std"],
            [[k.replace("_f1", "").upper(), fmt_stat(col(k), 4)] for k in ("rouge1_f1", "rouge2_f1", "rougeL_f1")]))
    report.notes += [
        f"{len(rows)} articles from {args.source}" + (f" ({args.month})" if args.source == "bbc" else "") + ".",
        "'extracted' counts triples that already passed the structural filters and the LLM verifier; "
        "'usable' additionally requires confidence >= the pipeline's MIN_CONFIDENCE.",
        "Reference summaries are the BBC article descriptions (one sentence), so ROUGE is low by design; "
        "use it to compare runs, not as an absolute quality score.",
    ]
    return report
