import datetime
import html
import re

from src.graph_builder import build_graph
from src.graph_insight import analyze_graph
from src.pipeline import AnalysisResult, render_graph_html
from src.report import build_report, summary_to_html

EVIL = "<img src=x onerror=alert(1)>"


def make_result(summary="• First point about **Acme** • Second point"):
    relations = [
        {"head": "Acme", "relation": "acquired", "tail": "Beta Corp", "head_type": "Organization",
         "tail_type": "Organization", "confidence": 0.9, "source_sentence": "Acme acquired Beta Corp."},
        {"head": EVIL, "relation": "attacked", "tail": "Acme", "head_type": "Person", "tail_type": "Organization",
         "confidence": 0.8, "source_sentence": f"{EVIL} attacked Acme."},
    ]
    G = build_graph(relations)
    return AnalysisResult(relations, analyze_graph(G), render_graph_html(G, lang="en"), summary, G,
                          {"elapsed": 75, "chars": 900, "truncated": True})


def test_report_contains_every_section_and_is_self_contained():
    result = make_result()
    page = build_report(result, result.html, name="demo", lang="en", palette="mint",
                        generated=datetime.datetime(2026, 1, 2, 3, 4))
    for anchor in ("#summary", "#graph", "#insights", "#relations"):
        assert f'href="{anchor}"' in page
    assert "Generated 2026-01-02 03:04" in page and "1:15 min" in page
    assert "<li>First point about <strong>Acme</strong></li>" in page
    assert "<iframe" in page and 'sandbox="allow-scripts"' in page
    assert page.count("<tr>") == 1 + len(result.relations)               # header + one row per relation
    assert "Only the first 900 characters" in page                        # truncation is reported
    assert not re.search(r"<script[^>]+src=", page)                       # no external scripts needed


def test_report_escapes_text_from_the_document():
    result = make_result(summary=f"- Evil {EVIL}")
    page = build_report(result, result.html, name='"><script>x</script>', lang="th")
    assert EVIL not in page and "<script>x</script>" not in page
    assert html.escape(EVIL) in page                                      # shown as text instead
    # the graph document is embedded as an attribute value: no unescaped quote can end it early
    srcdoc = re.search(r'srcdoc="([^"]*)"', page).group(1)
    assert html.unescape(srcdoc).lstrip().startswith("<html")


def test_report_follows_language_and_theme():
    result = make_result()
    th = build_report(result, result.html, lang="th", palette="rose", dark=True)
    assert 'lang="th"' in th and "รายงานกราฟความรู้" in th
    assert "--accent:#ff9cc4" in th and "--page:#0c0e15" in th           # dark rose palette
    assert "--accent:#cf4f86" in build_report(result, result.html, lang="en", palette="rose")


def test_summary_markdown_subset():
    assert summary_to_html("Intro line\n\n- a\n- **b**\n\nOuter") == (
        "<p>Intro line</p><ul><li>a</li><li><strong>b</strong></li></ul><p>Outer</p>")
    assert summary_to_html("1) one\n2. two") == "<ul><li>one</li><li>two</li></ul>"
    assert summary_to_html("Up 3.5% in 2) places") == "<p>Up 3.5% in 2) places</p>"
    assert "<script>" not in summary_to_html("<script>alert(1)</script>")


def test_app_offers_the_full_report_download(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.entity_resolution
    import src.pipeline

    fake = make_result()
    monkeypatch.setattr(src.entity_resolution, "load_embedder", lambda: None)
    monkeypatch.setattr(src.pipeline, "analyze_document", lambda *a, **k: fake)

    at = AppTest.from_file("app.py", default_timeout=90)
    at.session_state["lang"] = "en"
    at.run()
    labels = [b.label for b in at.button]
    at.button[labels.index(next(l for l in labels if l.startswith("sample")))].click().run()
    at.button[[b.label for b in at.button].index("Analyse")].click().run()
    assert not at.exception

    downloads = at.get("download_button")
    assert any(d.proto.label == "Download full report" for d in downloads)
