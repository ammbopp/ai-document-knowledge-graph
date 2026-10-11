"""Entity resolution: pairwise clustering quality across similarity thresholds and ablations."""
from itertools import combinations

from src.config import MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_OVERLAP, WINDOW_SIZE
from src.entity_resolution import EntityResolver
from src.pipeline import extract_chunks
from src.text_utils import create_sliding_windows, split_sentences

from .common import Report, fmt_stat, get_embedder, get_nlp, judge_json, load_articles, md_table, pct, prf

NAME = "resolution"
DESCRIPTION = "Entity clustering precision / recall / F1 (pairwise) per similarity threshold"

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
RULES_ONLY = 1.01  # embedding step can never fire: string rules (containment, surname) only
MAX_GOLD_ENTITIES = 80
MAX_TEXT_CHARS = 6000

GOLD_PROMPT = """Group the entity mentions below that refer to the SAME real-world person, organisation, place or thing in this article.

Article:
{text}

Mentions:
{mentions}

Rules: every mention must appear in exactly one group; a mention that matches nothing else forms its own group;
do not merge different things that merely share a word. Copy mentions exactly.
Answer with JSON: {{"groups": [["mention", "mention"], ["mention"]]}}"""


# ---------- pure helpers (unit-tested) ----------
def pairs_of(groups):
    """All unordered pairs of items that share a group."""
    return {frozenset(p) for members in groups for p in combinations(sorted(set(members)), 2)}


def pair_counts(gold_groups, predicted_groups):
    gold, pred = pairs_of(gold_groups), pairs_of(predicted_groups)
    return len(gold & pred), len(pred - gold), len(gold - pred)  # tp, fp, fn


def normalise_gold(groups, mentions):
    """
    Make LLM-written groups valid: keep only listed mentions (case-insensitive), first group wins on
    duplicates, and every unassigned mention becomes a singleton.
    """
    lookup = {m.lower(): m.lower() for m in mentions}
    assigned, result = set(), []
    for group in groups if isinstance(groups, list) else []:
        members = []
        for item in group if isinstance(group, list) else []:
            key = lookup.get(str(item).strip().lower())
            if key and key not in assigned:
                assigned.add(key)
                members.append(key)
        if members:
            result.append(members)
    result += [[m] for m in lookup if m not in assigned]
    return result


def predicted_groups(resolver, mentions):
    clusters = {}
    for m in mentions:
        clusters.setdefault(resolver.cluster_of(m), []).append(m.lower())
    return list(clusters.values())


# ---------- experiment ----------
def gold_groups(text, mentions):
    listing = "\n".join(f"- {m}" for m in mentions)
    data = judge_json(GOLD_PROMPT.format(text=text[:MAX_TEXT_CHARS], mentions=listing))
    return None if data is None else normalise_gold(data.get("groups"), mentions)


def run(args):
    nlp, embedder = get_nlp(), get_embedder()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)
    configs = [(f"threshold {t:.2f}", t) for t in THRESHOLDS] + [("string rules only", RULES_ONLY)]
    totals = {name: [0, 0, 0] for name, _ in configs + [("exact match only", None)]}
    report = Report(NAME, "Entity resolution evaluation")

    for i, article in enumerate(articles, 1):
        print(f"[resolution {i}/{len(articles)}] {article.id}")
        chunks = create_sliding_windows(split_sentences(nlp, article.text), WINDOW_SIZE, WINDOW_OVERLAP)
        relations = [r for rs in extract_chunks(chunks) for r in rs if r.get("confidence", 0) >= MIN_CONFIDENCE]
        occurrences = [(r[side], r.get(f"{side}_type", "Entity")) for r in relations for side in ("head", "tail")]
        mentions = list(dict.fromkeys(m for m, _ in occurrences))[:MAX_GOLD_ENTITIES]
        if len(mentions) < 3:
            print("  too few entities, skipping")
            continue
        gold = gold_groups(article.text, mentions)
        if gold is None:
            print("  gold grouping failed, skipping")
            continue
        listed = set(mentions)

        results = {"exact match only": [[m.lower()] for m in mentions]}
        for name, threshold in configs:
            resolver = EntityResolver(threshold=threshold, model=embedder)
            for mention, kind in occurrences:        # resolve in document order, like the pipeline
                resolver.resolve(mention, kind)
            results[name] = predicted_groups(resolver, [m for m in mentions if m in listed])
        for name, predicted in results.items():
            tp, fp, fn = pair_counts(gold, predicted)
            for k, v in enumerate((tp, fp, fn)):
                totals[name][k] += v
            p, r, f1 = prf(tp, fp, fn)
            report.rows.append({"article": article.id, "config": name, "mentions": len(mentions),
                                "gold_pairs": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                                "precision": p, "recall": r, "f1": f1})

    table = []
    for name in totals:
        tp, fp, fn = totals[name]
        p, r, f1 = prf(tp, fp, fn)
        per_article = [row["f1"] for row in report.rows if row["config"] == name]
        mark = " (production)" if name == f"threshold {RESOLVER_THRESHOLD:.2f}" else ""
        table.append([name + mark, pct(p), pct(r), pct(f1), fmt_stat(per_article, 1, pct=True)])
    report.add_section("Pairwise clustering quality (pooled over articles)", md_table(
        ["Configuration", "Precision", "Recall", "F1 (pooled)", "F1 per article (mean ± std)"], table))
    gold_pairs = totals["exact match only"][0] + totals["exact match only"][2]
    report.notes += [
        f"{len({r['article'] for r in report.rows})} articles with enough entities; {gold_pairs} gold same-entity pairs in total.",
        "Gold clusters are written by the LLM from the whole article and then validated; articles where the "
        "judge failed are skipped. Mentions are resolved in document order with a fresh resolver per article.",
        "'string rules only' disables the embedding step (containment and surname rules remain); "
        "'exact match only' never merges. Role linking (President -> person) is not part of this test.",
        "With few gold pairs the pooled numbers move a lot per article: raise --samples before drawing conclusions.",
    ]
    return report
