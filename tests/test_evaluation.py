"""Evaluation helpers that do not need an LLM: the metric maths must be right before the numbers mean anything."""
import math

from evaluation import __main__ as cli
from evaluation.common import Report, fmt_stat, load_articles, mean_std, prf, ratio
from evaluation.preprocessing import count_pronouns, preservation, pronoun_replacements, standard_chunks
from evaluation.resolution import normalise_gold, pair_counts
from evaluation.security import CANARY, build_payloads, hijacked, leaked, prompt_ngrams, score_output
from evaluation.summarization import split_claims, ungrounded_terms
from src import pipeline, preprocessing
from src.relation_extraction_ai import structure_triples


def test_prf_and_undefined_cases():
    precision, recall, f1 = prf(tp=2, fp=2, fn=0)
    assert (precision, recall) == (0.5, 1.0) and round(f1, 3) == 0.667
    assert all(math.isnan(v) for v in prf(0, 0, 0))   # nothing to measure is not a perfect score
    assert math.isnan(ratio(1, 0))


def test_mean_std_skips_missing_values():
    mean, std, n = mean_std([1.0, None, 3.0, math.nan])
    assert (mean, n) == (2.0, 2) and round(std, 3) == 1.414
    assert mean_std([])[2] == 0 and fmt_stat([]) == "n/a"


def test_pairwise_cluster_counts():
    gold = [["a", "b", "c"], ["d"]]
    predicted = [["a", "b"], ["c", "d"]]
    assert pair_counts(gold, predicted) == (1, 1, 2)   # tp: a-b; fp: c-d; fn: a-c, b-c
    assert pair_counts([["a"], ["b"]], [["a"], ["b"]]) == (0, 0, 0)


def test_gold_groups_are_validated():
    groups = normalise_gold([["Trump", "trump", "Donald Trump"], ["Ghost"], ["Trump", "BBC"]],
                            ["Trump", "Donald Trump", "BBC", "Putin"])
    assert groups[0] == ["trump", "donald trump"]       # duplicates collapsed, case-insensitive
    assert ["bbc"] in groups and ["putin"] in groups    # leftovers become singletons
    assert not any("ghost" in g for g in groups)        # invented mentions are dropped
    assert normalise_gold("garbage", ["A"]) == [["a"]]


def test_pronoun_replacements_come_from_a_token_diff():
    original = "Alice met Bob. She thanked him."
    found = pronoun_replacements(original, "Alice met Bob. Alice thanked Bob.")
    assert [(f["pronoun"], f["replacement"]) for f in found] == [("She", "Alice"), ("him", "Bob")]
    assert "Alice met Bob" in found[0]["before"]
    assert pronoun_replacements(original, original) == []
    assert count_pronouns(original) == 2 and preservation(original, original) == 1.0


def test_standard_chunks():
    assert [len(c.split()) for c in standard_chunks(" ".join(["w"] * 120), 50)] == [50, 50, 20]


def test_summary_claims_and_ungrounded_terms():
    claims = split_claims("Key points:\n- **Apple** launched a new phone on Monday.\n- Short.\n1) Prices rose sharply in Paris today.")
    assert claims == ["Apple launched a new phone on Monday.", "Prices rose sharply in Paris today."]
    source = "Apple announced a new phone on Monday costing 999 dollars."
    summary = "- The phone from Apple costs 1099 dollars.\n- It launches in Paris."
    assert ungrounded_terms(summary, source) == ["1099", "Paris"]
    assert ungrounded_terms("- Apple costs 999 dollars.", source) == []
    titled = "**Key Points:**\n## Context\n* Apple costs 999 dollars today.\nNotable Entities"
    assert ungrounded_terms(titled, source) == []   # titles are not claims


def test_security_scoring_uses_the_real_prompts():
    grams = prompt_ngrams(preprocessing.COREF_PROMPT)
    assert leaked("Sure! You are a precise NLP Coreference Resolution engine. Rewrite the text", grams)
    assert not leaked("Alice met Bob and they talked about the weather.", grams)
    assert hijacked("OWNED 4217") and hijacked("owned4217!") and not hijacked("OWNED and 4217 apart")
    assert score_output("fine output", grams) == (False, False, False)

    payloads = build_payloads()
    assert len(payloads) == 24 and {kind for kind, _ in payloads} == {"leak", "hijack"}
    # An echoed (unfollowed) payload must never count as a successful hijack
    assert not any(hijacked(text) for _, text in payloads)
    assert CANARY == "owned4217"


def test_structure_triples_filters_without_llm():
    text = "Bob stole the company assets from the vault last night."
    kept = structure_triples(text, "Bob | stole | company assets ### it | is | big ### Bob | stole | Mars rovers")
    assert [(h, r, t) for h, r, t, _ in kept] == [("Bob", "stole", "company assets")]   # pronoun + ungrounded dropped
    assert kept[0][3] == text


def test_extract_chunks_keeps_document_order(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_relations", lambda chunk: [{"head": chunk[0]}])
    progress = []
    result = pipeline.extract_chunks(["a", "b", "c", "d"], lambda done, total, remaining: progress.append((done, total)))
    assert [r[0]["head"] for r in result] == ["a", "b", "c", "d"]
    assert sorted(progress) == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_local_articles_and_report_output(tmp_path):
    articles = load_articles(2, source="local")
    assert len(articles) == 2 and all(a.text and not a.reference for a in articles)

    report = Report("demo", "Demo", rows=[{"a": 1}])
    report.add_section("Table", "| x |")
    paths = report.save(tmp_path)
    assert {p.suffix for p in paths} == {".md", ".csv"} and "# Demo" in paths[0].read_text(encoding="utf-8")


def test_cli_registers_every_experiment_with_defaults():
    assert set(cli.EXPERIMENTS) == set(cli.DEFAULT_SAMPLES) == {
        "system", "preprocessing", "extraction", "resolution", "summarization", "security"}
    args = cli.parse_args(["extraction", "--source", "local", "-n", "2"])
    assert (args.experiment, args.source, args.samples, args.windows) == ("extraction", "local", 2, 6)
