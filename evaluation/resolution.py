"""Entity resolution: pairwise precision / recall across similarity thresholds, judged pair by pair."""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from src.config import MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_OVERLAP, WINDOW_SIZE
from src.entity_resolution import EntityResolver
from src.pipeline import extract_chunks
from src.text_utils import create_sliding_windows, split_sentences

from .common import Report, as_bool, fmt_stat, get_embedder, get_nlp, judge_json, load_articles, md_table, pct, prf

NAME = "resolution"
DESCRIPTION = "Entity clustering precision / recall on hand-labelled mention pairs, per similarity threshold"

GOLD_PATH = Path(__file__).parent / "gold" / "resolution_pairs.json"
EMPTY = {"same": set(), "different": set(), "ambiguous": set()}   # an article without labels

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
RULES_ONLY = 1.01          # embedding step can never fire: string rules (containment, surname) only
MAX_MENTIONS = 80
MAX_POOL = 160             # judged pairs per article
POOL_SIMILARITY = 0.60     # embedding similarity that makes a pair worth judging even if no config merged it
_STOP = {"the", "of", "and", "in", "on", "at", "for", "to", "a", "an", "mr", "mrs", "ms", "dr"}

PAIR_PROMPT = """Decide whether two mentions from a news article refer to the SAME real-world entity.

Mention A: "{a}"
Context of A: "{ctx_a}"

Mention B: "{b}"
Context of B: "{ctx_b}"

Answer true only if A and B are the same person, organisation, place or thing (a different wording, a surname, an
abbreviation or a plural of the same thing counts). Answer false for related but different things (a person and
their company, a city and its country, a part and the whole, two different people).
Answer with JSON: {{"same": true}} or {{"same": false}}"""


# ---------- pure helpers (unit-tested) ----------
def pairs_of(groups):
    """All unordered pairs of items that share a group."""
    return {frozenset(p) for members in groups for p in combinations(sorted(set(members)), 2)}


def pair_counts(gold_groups, predicted_groups):
    gold, pred = pairs_of(gold_groups), pairs_of(predicted_groups)
    return len(gold & pred), len(pred - gold), len(gold - pred)  # tp, fp, fn


def content_tokens(mention):
    return {w for w in re.findall(r"[a-z0-9]+", mention.lower()) if w not in _STOP and len(w) > 1}


def lexical_overlap(a, b):
    """True if the mentions share a content word ("Welsh Ambulance Service" / "ambulance service")."""
    return bool(content_tokens(a) & content_tokens(b))


def score_against_judgements(predicted_pairs, same_pairs, different_pairs):
    """
    Counts over the judged pool only: tp = predicted and judged same; fp = predicted and judged different;
    fn = judged same but not predicted. Pairs the judge could not decide are in neither set.
    """
    return (len(predicted_pairs & same_pairs), len(predicted_pairs & different_pairs),
            len(same_pairs - predicted_pairs))


def predicted_groups(resolver, mentions):
    clusters = {}
    for m in mentions:
        clusters.setdefault(resolver.cluster_of(m), []).append(m.lower())
    return list(clusters.values())


def candidate_pool(mentions, config_pairs, embeddings=None, similarity=POOL_SIMILARITY, limit=MAX_POOL):
    """
    Pairs worth judging: everything any configuration merged, plus pairs that look alike (shared word or
    close embeddings). Merged pairs come first so the cap never drops a prediction.
    """
    merged = set().union(*config_pairs) if config_pairs else set()
    extra = set()
    lowered = [m.lower() for m in mentions]
    for i, j in combinations(range(len(lowered)), 2):
        pair = frozenset((lowered[i], lowered[j]))
        if len(pair) < 2 or pair in merged:
            continue
        if lexical_overlap(lowered[i], lowered[j]) or (
                embeddings is not None and float(embeddings[i] @ embeddings[j]) >= similarity):
            extra.add(pair)
    ordered = sorted(merged, key=sorted) + sorted(extra, key=sorted)
    return ordered[:limit]


def load_gold(path=GOLD_PATH):
    """{article_id: {"same": {frozenset}, "different": {frozenset}, "ambiguous": {frozenset}}} from the labels file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["articles"]
    return {aid: {kind: {frozenset(p) for p in entry.get(kind, [])} for kind in ("same", "different", "ambiguous")}
            for aid, entry in raw.items()}


def label_pairs(pool, labels):
    """Split a candidate pool by the hand labels -> (same, different, ambiguous, unlabelled)."""
    same = {p for p in pool if p in labels["same"]}
    different = {p for p in pool if p in labels["different"]}
    ambiguous = {p for p in pool if p in labels["ambiguous"]}
    return same, different, ambiguous, [p for p in pool if p not in same | different | ambiguous]


# ---------- experiment ----------
def judge_pair(a, b, context):
    verdict = as_bool((judge_json(PAIR_PROMPT.format(
        a=a, ctx_a=context.get(a, ""), b=b, ctx_b=context.get(b, ""))) or {}).get("same"))
    return verdict


def run(args):
    nlp, embedder = get_nlp(), get_embedder()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)
    configs = [(f"threshold {t:.2f}", t) for t in THRESHOLDS] + [("string rules only", RULES_ONLY)]
    names = [n for n, _ in configs] + ["exact match only"]
    totals = {n: [0, 0, 0] for n in names}
    report = Report(NAME, "Entity resolution evaluation")
    gold = load_gold(args.gold or GOLD_PATH)
    judged_total = failed_total = same_total = ambiguous_total = unlabelled_total = 0

    for i, article in enumerate(articles, 1):
        print(f"[resolution {i}/{len(articles)}] {article.id}")
        chunks = create_sliding_windows(split_sentences(nlp, article.text), WINDOW_SIZE, WINDOW_OVERLAP)
        relations = [r for rs in extract_chunks(chunks) for r in rs if r.get("confidence", 0) >= MIN_CONFIDENCE]
        occurrences = [(r[side], r.get(f"{side}_type", "Entity"), r.get("source_sentence", ""))
                       for r in relations for side in ("head", "tail")]
        mentions = list(dict.fromkeys(m for m, _, _ in occurrences))[:MAX_MENTIONS]
        if len(mentions) < 3:
            print("  too few entities, skipping")
            continue
        context = {}
        for m, _, sentence in occurrences:
            context.setdefault(m.lower(), sentence)
        listed = set(mentions)

        predictions = {"exact match only": set()}
        for name, threshold in configs:
            resolver = EntityResolver(threshold=threshold, model=embedder)
            for mention, kind, _ in occurrences:         # resolve in document order, like the pipeline
                resolver.resolve(mention, kind)
            predictions[name] = pairs_of(predicted_groups(resolver, [m for m in mentions if m in listed]))

        vectors = np.asarray(embedder.encode([m.lower() for m in mentions]), dtype=float)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        pool = candidate_pool(mentions, list(predictions.values()), vectors)
        same, different, ambiguous, unlabelled = label_pairs(pool, gold.get(article.id, EMPTY))
        if args.llm_judge:                       # optional: judge unlabelled pairs with the LLM (lenient; see notes)
            for pair in unlabelled:
                a, b = sorted(pair)
                verdict = judge_pair(a, b, context)
                if verdict is None:
                    failed_total += 1
                else:
                    (same if verdict else different).add(pair)
            unlabelled = []
        judged_total += len(same) + len(different)
        same_total += len(same)
        ambiguous_total += len(ambiguous)
        unlabelled_total += len(unlabelled)
        print(f"  {len(mentions)} mentions, {len(pool)} candidate pairs: {len(same)} same, {len(different)} different, "
              f"{len(ambiguous)} ambiguous, {len(unlabelled)} unlabelled")

        for name in names:
            tp, fp, fn = score_against_judgements(predictions[name], same, different)
            for k, v in enumerate((tp, fp, fn)):
                totals[name][k] += v
            p, r, f1 = prf(tp, fp, fn)
            report.rows.append({"article": article.id, "config": name, "mentions": len(mentions),
                                "pairs_judged": len(same) + len(different), "same_pairs": len(same),
                                "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1})

    table = []
    for name in names:
        tp, fp, fn = totals[name]
        p, r, f1 = prf(tp, fp, fn)
        per_article = [row["f1"] for row in report.rows if row["config"] == name]
        mark = " (production)" if name == f"threshold {RESOLVER_THRESHOLD:.2f}" else ""
        table.append([name + mark, pct(p), pct(r), pct(f1), fmt_stat(per_article, 1, pct=True), f"{tp}/{fp}/{fn}"])
    report.add_section("Pairwise clustering quality (pooled over articles)", md_table(
        ["Configuration", "Precision", "Recall", "F1 (pooled)", "F1 per article (mean ± std)", "TP / FP / FN"], table))
    source = ("hand labels in evaluation/gold/resolution_pairs.json" +
              (" plus LLM verdicts for unlabelled pairs (the LLM judge is lenient: it calls most look-alikes 'same')"
               if args.llm_judge else ""))
    report.notes += [
        f"{len({r['article'] for r in report.rows})} articles with enough entities; of the candidate pairs, "
        f"{judged_total} were scored ({same_total} same entity), {ambiguous_total} labelled ambiguous and "
        f"{unlabelled_total} unlabelled (both left out)" + (f", {failed_total} judge failures" if args.llm_judge else "") + ".",
        f"Labels: {source}.",
        "Candidate pairs = every pair any configuration merged, plus pairs sharing a content word or with embedding "
        f"similarity >= {POOL_SIMILARITY}. Precision is measured on pairs a configuration merged; recall is relative "
        "to the same-entity pairs in this pool, so same-entity pairs that look nothing alike are not counted.",
        "'string rules only' disables the embedding step (containment and surname rules remain); "
        "'exact match only' never merges. Role linking (President -> person) is not part of this test.",
        "The labelled set is small; treat differences of a few points between thresholds as noise.",
    ]
    return report
