"""Relation-extraction gates: raw LLM output vs structural filters vs structural + LLM verifier."""
from concurrent.futures import ThreadPoolExecutor

from src.config import LLM_WORKERS, MIN_CONFIDENCE, WINDOW_OVERLAP, WINDOW_SIZE
from src.relation_extraction_ai import ask_llm, evaluate_triplet_cloze, parse_triples, structure_triples
from src.text_utils import create_sliding_windows, split_sentences

from .common import Report, as_bool, get_nlp, judge_json, load_articles, md_table, pct, ratio

NAME = "extraction"
DESCRIPTION = "Triple precision / hallucination rate for raw LLM output vs the filter + verifier gates"

JUDGE_PROMPT = """You are a strict fact checker for a knowledge graph. Check ONE extracted triplet against its source text.

Source text: "{source}"
Triplet: [{head}] --({relation})--> [{tail}]

Answer these questions:
- supported: is the triplet stated or clearly implied by the text, with the right direction (the subject really performs the relation on the object)?
- speech: is the RELATION PHRASE itself a speech/reporting verb ("said", "told", "announced", "claimed")? A factual
  relation that merely appears inside reported speech (e.g. "impose" in "X said he will impose tariffs") is NOT speech.
- invented: does the triplet contain names or facts that are NOT in the text?
- standalone: are BOTH [{head}] and [{tail}] concrete, self-contained things (a person, organisation, place, date, object or specific
  noun phrase) that a reader would understand without the text? Answer false if either is a pronoun ("he", "this"), a
  vague word ("then", "it"), or a sentence fragment.

Answer with JSON: {{"supported": true/false, "speech": true/false, "invented": true/false, "standalone": true/false}}"""

GROUPS = ["raw LLM output", "+ structural filters", "+ LLM verifier (full)"]


def process_window(window):
    """Run one window through the gates; each group holds (head, relation, tail, source) tuples."""
    output = ask_llm(window)
    raw = [(h, r, t, window) for h, r, t in parse_triples(output)]
    structural = [(h, r, t, ev) for h, r, t, ev in structure_triples(window, output)]
    verified = []
    for h, r, t, ev in structural:
        ok, confidence = evaluate_triplet_cloze(h, r, t, ev)
        if ok and confidence >= MIN_CONFIDENCE:
            verified.append((h, r, t, ev))
    return {GROUPS[0]: raw, GROUPS[1]: structural, GROUPS[2]: verified}


def judge_triple(triple):
    head, relation, tail, source = triple
    data = judge_json(JUDGE_PROMPT.format(source=source, head=head, relation=relation, tail=tail)) or {}
    supported, speech, invented, standalone = (
        as_bool(data.get(k)) for k in ("supported", "speech", "invented", "standalone"))
    if None in (supported, speech, invented, standalone):
        return None
    return {"correct": supported and standalone and not speech, "speech": speech, "invented": invented,
            "standalone": standalone}


def run(args):
    nlp = get_nlp()
    articles = load_articles(args.samples, source=args.source, month=args.month, offset=args.offset)
    windows = []
    for article in articles:
        sentences = split_sentences(nlp, article.text)
        for w in create_sliding_windows(sentences, WINDOW_SIZE, WINDOW_OVERLAP)[:args.windows]:
            windows.append((article.id, w))
    print(f"[extraction] {len(articles)} articles, {len(windows)} windows")

    with ThreadPoolExecutor(max_workers=max(1, LLM_WORKERS)) as pool:
        processed = list(pool.map(lambda aw: process_window(aw[1]), windows))
        unique = list({t for p in processed for g in GROUPS for t in p[g]})
        print(f"[extraction] judging {len(unique)} distinct triples")
        verdicts = dict(zip(unique, pool.map(judge_triple, unique)))

    report = Report(NAME, "Extraction gate evaluation")
    table, correct_by_group = [], {}
    for group in GROUPS:
        triples = [t for p in processed for t in p[group]]
        judged = [verdicts[t] for t in triples if verdicts[t] is not None]
        correct = sum(v["correct"] for v in judged)
        correct_by_group[group] = correct
        row = {"group": group, "triples": len(triples), "judged": len(judged), "correct": correct,
               "precision": ratio(correct, len(judged)),
               "hallucination_rate": ratio(sum(v["invented"] for v in judged), len(judged)),
               "speech_rate": ratio(sum(v["speech"] for v in judged), len(judged)),
               "vague_entity_rate": ratio(sum(not v["standalone"] for v in judged), len(judged))}
        report.rows.append(row)
    raw_correct = correct_by_group[GROUPS[0]]
    for row in report.rows:
        row["retention"] = ratio(row["correct"], raw_correct)
        table.append([row["group"], row["triples"], row["judged"], pct(row["precision"]),
                      pct(row["hallucination_rate"]), pct(row["speech_rate"]), pct(row["vague_entity_rate"]),
                      pct(row["retention"])])

    report.add_section("Gate comparison (pooled over all windows)", md_table(
        ["Pipeline stage", "Triples", "Judged", "Precision", "Hallucination rate", "Speech/opinion rate",
         "Vague-entity rate", "Correct triples kept vs raw"], table))
    failed = sum(1 for v in verdicts.values() if v is None)
    report.notes += [
        f"{len(articles)} articles, first {args.windows} sliding windows each ({len(windows)} windows).",
        "Precision = judged supported (correct direction), not a speech wrapper, and both entities are "
        "standalone (no pronouns or fragments). 'Kept vs raw' is "
        "correct triples in the group divided by correct raw triples; entity cleaning means strings differ, "
        "so treat it as an approximate recall proxy.",
        f"Judge failures excluded from the rates: {failed}.",
        "The extractor and the judge are the same model: expect optimistic precision for every group.",
    ]
    return report
