from src.grounding import highlight, summary_flags, ungrounded_terms
from src.pipeline import AnalysisResult
from src.report import build_report, summary_to_html

SOURCE = ("Acme Corp announced on Monday that it will pay 1,099 dollars per seat. Chief executive Dana Reyes said "
          "American customers in Mexico and Brazil were the first to sign up, and tariffs rose 3.5% in 2024.")


def test_invented_names_numbers_and_sentence_starts_are_flagged():
    summary = ("- Acme Corp will charge 1099 dollars and 2500 for premium.\n"
               "- Moscow welcomed the plan, according to Dana Reyes.\n"
               "- Revenue rose 7.2% after Beta Industries joined.")
    assert ungrounded_terms(summary, SOURCE) == ["2500", "7.2", "Beta", "Industries", "Moscow"]


def test_inflections_and_formatting_are_not_false_alarms():
    summary = "- The Americans in Mexican markets saw tariff increases of 3.5% in 2024.\n- Acme Corp's seats cost 1,099 dollars."
    assert ungrounded_terms(summary, SOURCE) == []
    assert ungrounded_terms("- Peters visited", "Petersburg is a city") == ["Peters"]   # different name, not an inflection
    assert ungrounded_terms("", SOURCE) == [] and ungrounded_terms("- Anything Goes", "") == []


def test_headings_are_ignored():
    assert ungrounded_terms("**Key Points:**\n## Context\n* Acme Corp pays 1,099 dollars.\nNotable Entities", SOURCE) == []


def test_highlight_wraps_text_but_never_tags():
    html = summary_to_html("- Moscow welcomed 2500 <i>units</i> & Moscow's 'plan'")
    marked = highlight(html, ["Moscow", "2500", "amp", "lt", "x27"], "Not in source")
    assert marked.count('<mark class="ungrounded" title="Not in source">Moscow</mark>') == 2
    assert '>2500</mark>' in marked
    assert "<li>" in marked and "&lt;i&gt;" in marked and "&amp;" in marked and "&#x27;" in marked   # entities intact
    assert highlight("<p>Nothing</p>", []) == "<p>Nothing</p>"


def test_flags_are_cached_on_the_result_and_default_to_clean_without_source():
    result = AnalysisResult([], {}, "", "- Moscow welcomed it.", None, {}, SOURCE)
    assert summary_flags(result) == ["Moscow"] and result.stats["ungrounded"] == ["Moscow"]
    result.stats["ungrounded"] = ["cached"]
    assert summary_flags(result) == ["cached"]
    assert summary_flags(AnalysisResult([], {}, "", "- Moscow welcomed it.", None, {})) == []   # no source to check against


def test_report_marks_flagged_terms_and_reports_clean_summaries():
    flagged = AnalysisResult([], {}, "", "- Acme Corp pays 1,099 dollars. - Moscow agreed.", None, {}, SOURCE)
    page = build_report(flagged, "<html></html>", lang="en")
    assert "<mark class=\"ungrounded\"" in page and ">Moscow</mark>" in page and "class='warn'" in page

    clean = AnalysisResult([], {}, "", "- Acme Corp pays 1,099 dollars.", None, {}, SOURCE)
    page = build_report(clean, "<html></html>", lang="en")
    assert "class='ok'" in page and "<mark" not in page.split("<section id=\"graph\"")[0]


def test_labels_and_enumerators_are_not_facts():
    summary = ("**Note:** The summary only includes information from the text.\n"
               "Key takeaway: Acme Corp pays 1,099 dollars per seat.\n"
               "Topic Shift 3 covers pricing. 2. Acme Corp announced it on Monday.\n"
               "Costs rose 3.5% in 2024 and 12 offices opened.")
    assert ungrounded_terms(summary, SOURCE) == ["12"]      # only a genuinely new number is left
    assert ungrounded_terms("Moscow: approved the deal.", SOURCE) == ["Moscow"]   # a name before ':' is still checked


def test_bold_labels_titles_plurals_spellings_and_number_words():
    source = ("Izzett has raced nine or 10 buses. Local GP practices and the Latinisation of names were discussed. "
              "Acme Corp pays 1,099 dollars.")
    summary = ("**Fitness Innovation: Racing Against Buses**\n"
               "• **Not for Beginners:** However, he raced 9-10 buses.\n"
               "* Present: Overall, GPs discussed the Latinization of names.\n"
               "- The deal was signed in 1925 by Acme Corp.")
    assert ungrounded_terms(summary, source) == ["1925"]     # only the invented year survives
    # a bold name that is part of the sentence is still checked
    assert ungrounded_terms("**Moscow** welcomed the plan.", SOURCE) == ["Moscow"]
