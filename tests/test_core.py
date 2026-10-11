import numpy as np
import pytest

from src.entity_resolution import EntityResolver, resolve_relations
from src.graph_builder import build_graph
from src.graph_insight import analyze_graph
from src.relation_extraction_ai import (
    infer_entity_type, is_hallucination, is_valid_entity, nlp, parse_triples,
)
from src.text_utils import create_sliding_windows, split_by_chars
from src.web_fetch import decode_upload, validate_public_url


class FakeEmbedder:
    """Deterministic bag-of-characters embedder (no model download)."""

    def encode(self, texts):
        out = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out


def rel(h, r, t, ht="Person", tt="Organization"):
    return {"head": h, "relation": r, "tail": t, "head_type": ht, "tail_type": tt,
            "source_sentence": f"{h} {r} {t}.", "confidence": 1.0}


def test_parse_triples_handles_missing_hashes():
    text = "Bob | stole | assets\nAlice | hired | Bob ### Eve | met | Carl"
    assert parse_triples(text) == [
        ("Bob", "stole", "assets"), ("Alice", "hired", "Bob"), ("Eve", "met", "Carl")]


def test_entity_filters():
    assert is_valid_entity("Elon Musk")
    assert is_valid_entity("O'Brien")
    assert not is_valid_entity("he")
    assert not is_valid_entity("15")
    assert not is_valid_entity("is the best")


def test_hallucination_check_needs_whole_words():
    src = "Elon Musk leads SpaceX."
    assert not is_hallucination("Elon Musk", "SpaceX", src)
    assert is_hallucination("Jeff Bezos", "SpaceX", src)
    assert is_hallucination("a", "SpaceX", src)


def test_infer_entity_type():
    doc = nlp("Elon Musk joined Google in California.")
    assert infer_entity_type("Elon Musk", doc) == "Person"
    assert infer_entity_type("Google", doc) == "Organization"
    assert infer_entity_type("rockets", doc) == "Entity"


def test_sliding_windows_and_char_split():
    assert create_sliding_windows(["a", "b", "c", "d"], 3, 1) == ["a b c", "c d"]
    pieces = split_by_chars("word " * 100, 50)
    assert all(len(p) <= 50 for p in pieces) and " ".join(pieces).split() == ["word"] * 100


def test_resolver_is_per_instance_and_keeps_both_directions():
    r1 = EntityResolver(model=FakeEmbedder())
    assert r1.resolve("Elon Musk", "Person") == "Elon Musk"
    assert r1.resolve("Musk", "Person") == "Elon Musk"
    r2 = EntityResolver(model=FakeEmbedder())
    assert r2.resolve("Musk", "Person") == "Musk"  # nothing leaks from r1

    rels = [rel("Musk", "founded", "SpaceX"), rel("SpaceX", "employs", "Musk", "Organization", "Person"),
            rel("Elon Musk", "leads", "SpaceX"), rel("Elon Musk", "founded", "SpaceX")]
    out = resolve_relations(rels, EntityResolver(model=FakeEmbedder()))
    # reverse direction + different relation kept, exact duplicate dropped, names canonical
    assert {(r["head"], r["relation"], r["tail"]) for r in out} == {
        ("Elon Musk", "founded", "SpaceX"), ("SpaceX", "employs", "Elon Musk"),
        ("Elon Musk", "leads", "SpaceX")}


def test_resolver_type_conflict_not_merged():
    r = EntityResolver(model=FakeEmbedder())
    r.resolve("Apple", "Organization")
    assert r.resolve("Apple Pie", "Work") != "Apple"


def test_graph_keeps_parallel_relations_and_insights_ignore_virtual_edges():
    G = build_graph([rel("A Corp", "buys", "B Corp"), rel("A Corp", "sues", "B Corp"),
                     rel("B Corp", "hires", "A Corp"), rel("X Inc", "owns", "Y Inc")])
    assert G.number_of_edges("A Corp", "B Corp") == 2
    assert any(d.get("type") == "virtual" for _, _, d in G.edges(data=True))
    ins = analyze_graph(G)
    assert ins["total_relations"] == 4
    assert ins["num_connected_components"] == 2  # virtual link does not hide the split


def test_main_entity_is_deterministic():
    rels = [rel("Beta", "r", "Alpha"), rel("Alpha", "r", "Gamma")]
    tops = {max(build_graph(rels).nodes(data="group"), key=lambda x: x[1] == "Hub")[0] for _ in range(5)}
    assert len(tops) == 1


@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://localhost:8080", "ftp://example.com",
                                 "http://169.254.169.254/latest/meta-data"])
def test_url_validation_blocks_internal(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_decode_upload_fallbacks():
    assert decode_upload("héllo".encode("utf-8")) == "héllo"
    assert decode_upload("สวัสดี".encode("cp874")) == "สวัสดี"


def test_clean_entity_name_strips_possessives():
    from src.relation_extraction_ai import clean_entity_name
    assert clean_entity_name("Donald Trump's previous import taxes") == "previous import taxes"
    assert clean_entity_name("Trump's") == "Trump"
    assert clean_entity_name("the workers' union") == "union"
    assert clean_entity_name("O'Brien") == "O'Brien"


def test_normalize_passive_swaps_arguments():
    from src.relation_extraction_ai import normalize_passive
    assert normalize_passive("White House", "contacted by", "BBC") == ("BBC", "contacted", "White House")
    assert normalize_passive("Book", "was written by", "Alice") == ("Alice", "written", "Book")
    assert normalize_passive("Law", "struck down by", "Court") == ("Court", "struck down", "Law")
    # active forms stay untouched
    assert normalize_passive("Bob", "stood by", "Alice") == ("Bob", "stood by", "Alice")
    assert normalize_passive("Bob", "contacted", "Alice") == ("Bob", "contacted", "Alice")
    assert normalize_passive("WH", "has been contacted by", "BBC") == ("BBC", "contacted", "WH")


def test_split_trailing_date():
    from src.relation_extraction_ai import split_trailing_date
    assert split_trailing_date("force on Tuesday, 24 February") == ("force", "on", "Tuesday, 24 February")
    assert split_trailing_date("Tuesday, 24 February") is None  # pure date: nothing to split
    assert split_trailing_date("global tariffs") is None
    assert is_valid_entity("2026") and not is_valid_entity("15")


def test_resolver_prefers_frequent_form_and_skips_generic_containment():
    r = EntityResolver(model=FakeEmbedder())
    for name in ["Cristiano Ronaldo", "Cristiano Ronaldo", "departure of Cristiano Ronaldo"]:
        r.resolve(name, "Person")
    assert r.canonical("departure of Cristiano Ronaldo") == "Cristiano Ronaldo"
    r2 = EntityResolver(threshold=1.1, model=FakeEmbedder())
    r2.resolve("petition against data centre", "Entity")
    assert r2.resolve("data centre", "Entity") == "data centre"  # not merged by containment


def test_speech_verbs_and_coref_echo_rejected(monkeypatch):
    from src import preprocessing
    from src.relation_extraction_ai import is_valid_relation
    assert not is_valid_relation("told") and not is_valid_relation("said that")
    assert is_valid_relation("announced on")
    echoed = "CRITICAL SAFETY RULES: 1. blah <rewritten>x</rewritten>"
    monkeypatch.setattr(preprocessing, "generate", lambda *a, **k: echoed)
    assert preprocessing._resolve_chunk("original chunk here") == "original chunk here"
    monkeypatch.setattr(preprocessing, "generate", lambda *a, **k: "Bob met Alice.\n")
    assert preprocessing._resolve_chunk("He met Alice.") == "Bob met Alice."


def test_bare_passive_and_role_linking():
    from src.relation_extraction_ai import is_bare_passive
    assert is_bare_passive("has been suspended") and not is_bare_passive("was subject to")
    assert not is_bare_passive("contacted by") and not is_bare_passive("suspended")
    rels = [rel("President", "started war with", "Iran", "Entity", "Location"),
            rel("Donald Trump", "visited", "Iran", "Person", "Location")]
    out = resolve_relations(rels, EntityResolver(model=FakeEmbedder()))
    assert {r["head"] for r in out} == {"Donald Trump"} or {r["head"] for r in out} == {"President"}
    assert len({r["head"] for r in out}) == 1


def test_unknown_capitalised_names_are_valid_entities():
    assert is_valid_entity("Anthropic")


def test_surname_and_relation_key_dedup():
    from src.relation_extraction_ai import relation_key
    r = EntityResolver(threshold=1.1, model=FakeEmbedder())
    r.resolve("Koos Bekker", "Person")
    assert r.resolve("Bekker", "Entity") == "Koos Bekker"
    assert relation_key("will be causing") == relation_key("cause") or relation_key("causing") == relation_key("cause")
    assert relation_key("included on") == relation_key("will be included on")


def test_direction_check_parses_letter_not_substring(monkeypatch):
    from src import relation_extraction_ai as rea
    answers = iter(["The answer is A", "B"])
    monkeypatch.setattr(rea, "generate", lambda *a, **k: next(answers))
    assert rea.check_direction("Bob", "met", "Alice", "Bob met Alice.")
    answers = iter(["B", "A"])
    assert not rea.check_direction("Bob", "met", "Alice", "Alice met Bob.")


def test_surname_merges_despite_type_mismatch():
    r = EntityResolver(threshold=1.1, model=FakeEmbedder())
    r.resolve("Koos Bekker", "Person")
    r.resolve("Bekker", "Location")
    assert r.canonical("Bekker") == r.canonical("Koos Bekker")


def test_pipeline_helpers():
    from src.pipeline import estimate_seconds, truncate_at_sentence
    text = "First sentence here. Second sentence here. Third one is longer than the limit."
    cut = truncate_at_sentence(text, 45)
    assert cut.endswith(".") and len(cut) <= 45 and truncate_at_sentence(text, 500) == text
    assert estimate_seconds("x" * 1200) > estimate_seconds("x" * 120) >= 1


def test_ollama_health_reports_missing_model(monkeypatch):
    from src import llm

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "mistral:latest"}]}

    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp())
    ok, msg = llm.health()
    assert not ok and "ollama pull" in msg
    monkeypatch.setattr(llm, "LLM_MODEL", "mistral")
    assert llm.health()[0]

    def boom(*a, **k): raise ConnectionError
    monkeypatch.setattr(llm.requests, "get", boom)
    assert not llm.health()[0]


def test_app_renders_and_loads_sample():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    labels = [b.label for b in at.button]
    sample = "ใช้"
    at.button[labels.index(sample)].click().run()
    assert not at.exception and any(m.label == "จำนวนคำ" for m in at.metric)


def test_palettes_are_valid_hex_and_drive_css_vars():
    import re
    from src.themes import DEFAULT_PALETTE, PALETTES, accent, css_vars
    assert DEFAULT_PALETTE in PALETTES and len(PALETTES) >= 5
    for light, dark in PALETTES.values():
        assert re.fullmatch(r"#[0-9a-f]{6}", light) and re.fullmatch(r"#[0-9a-f]{6}", dark)
    assert accent(DEFAULT_PALETTE) != accent(DEFAULT_PALETTE, dark=True)
    assert "--accent:" + accent(DEFAULT_PALETTE) in css_vars(DEFAULT_PALETTE)
    assert css_vars("unknown-name")  # falls back to the default palette
