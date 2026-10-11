"""One self-contained HTML report: summary, interactive graph, insights and the relations table."""
import datetime
import html
import re

from .graph_view import NODE_TYPES, node_colors
from .grounding import highlight, summary_flags
from .i18n import t, type_names
from .themes import accent, css_vars

CSS = """
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font-family:'Prompt','Noto Sans Thai',system-ui,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 64px}
header.top{padding:44px 0 18px}
header.top h1{margin:0 0 6px;font-size:2rem;font-weight:500;letter-spacing:-.02em}
header.top .meta{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;color:var(--muted);font-size:.86rem}
.pill{display:inline-flex;align-items:center;gap:.4rem;padding:.15rem .75rem;border-radius:999px;background:var(--card);
  border:1px solid var(--line);font-size:.8rem;color:var(--ink)}
nav{position:sticky;top:0;z-index:20;display:flex;gap:.4rem;flex-wrap:wrap;padding:10px 0;margin-bottom:10px;
  background:var(--page);border-bottom:1px solid var(--line)}
nav a{color:var(--ink);text-decoration:none;padding:.35rem 1rem;border-radius:999px;font-size:.88rem;border:1px solid transparent}
nav a:hover{border-color:var(--ink)}
section{margin:34px 0;scroll-margin-top:60px}
section>h2{font-size:1.35rem;font-weight:500;margin:0 0 14px;letter-spacing:-.01em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:18px 20px;box-shadow:0 14px 34px -26px var(--shadow)}
.card .k{font-size:.8rem;color:var(--muted)}
.card .v{font-size:2rem;font-weight:500;letter-spacing:-.02em}
.card .hint{font-size:.78rem;color:var(--muted)}
.prose{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px 26px}
.prose p{margin:.2rem 0 .8rem}.prose ul{margin:.2rem 0 .4rem;padding-left:1.3rem}.prose li{margin:.3rem 0}
.note{font-size:.82rem;color:var(--muted);margin-top:10px}
mark.ungrounded{background:color-mix(in srgb,var(--warm) 55%,transparent);color:inherit;border-bottom:2px solid var(--warm);
  border-radius:4px;padding:0 .18em}
.warn,.ok{border-radius:14px;padding:10px 14px;margin:0 0 14px;font-size:.9rem}
.warn{background:color-mix(in srgb,var(--warm) 28%,var(--card));border:1px solid var(--warm)}
.ok{background:var(--card);border:1px solid var(--line);color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 10px}
.legend span{display:inline-flex;align-items:center;gap:.45rem;font-size:.82rem;padding:.2rem .75rem;border-radius:999px;
  background:var(--card);border:1px solid var(--line)}
.legend i{width:.6rem;height:.6rem;border-radius:50%;display:inline-block}
iframe.graph{width:100%;height:680px;border:1px solid var(--line);border-radius:20px;background:var(--card);display:block}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.card h3{margin:.1rem 0 .1rem;font-size:1.25rem;font-weight:500;word-break:break-word}
.bars{display:grid;gap:8px;margin-top:6px}
.bar{display:grid;grid-template-columns:minmax(120px,30%) 1fr 2.5em;align-items:center;gap:10px;font-size:.88rem}
.bar .track{height:12px;border-radius:999px;background:var(--line);overflow:hidden}
.bar .fill{display:block;height:100%;background:var(--accent);border-radius:999px}
.bar .n{color:var(--muted);text-align:right}
.groups{margin:.3rem 0 0;padding:0;list-style:none;display:grid;gap:8px}
.groups li{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 16px;font-size:.92rem}
.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.tools input{flex:1;min-width:220px;padding:.6rem 1rem;border-radius:999px;border:1px solid var(--line);background:var(--card);
  color:var(--ink);font:inherit;font-size:.9rem}
.tools input:focus{outline:2px solid var(--accent);outline-offset:1px}
.tools .count{color:var(--muted);font-size:.84rem}
.tablebox{background:var(--card);border:1px solid var(--line);border-radius:20px;overflow:auto;max-height:640px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th{position:sticky;top:0;background:var(--card);text-align:left;font-weight:500;color:var(--muted);padding:12px 14px;
  border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:10px 14px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.src{color:var(--muted);min-width:260px}
td.conf{white-space:nowrap}
tr[hidden]{display:none}
footer{margin-top:48px;color:var(--muted);font-size:.8rem;text-align:center}
@media print{nav,.tools{display:none}iframe.graph{height:560px}.tablebox{max-height:none;overflow:visible}body{background:#fff}}
"""

SCRIPT = """
(function () {
  var input = document.getElementById('filter'), rows = document.querySelectorAll('#relations tbody tr'),
      count = document.getElementById('filter-count'), total = rows.length, template = count.dataset.template;
  function update() {
    var q = input.value.trim().toLowerCase(), shown = 0;
    rows.forEach(function (row) {
      var match = !q || row.textContent.toLowerCase().indexOf(q) !== -1;
      row.hidden = !match;
      if (match) shown++;
    });
    count.textContent = template.replace('{shown}', shown).replace('{total}', total);
  }
  input.addEventListener('input', update);
  update();
})();
"""


def _e(value):
    return html.escape(str(value), quote=True)


def summary_to_html(text):
    """Small, safe Markdown subset (paragraphs, bullet lists, **bold**, headings) for LLM summaries."""
    text = re.sub(r"\s*•\s*", "\n- ", text.strip())   # LLMs often put "• a • b" on one line
    out, items = [], []

    def inline(s):
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _e(s))

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in text.splitlines():
        line = line.strip()
        bullet = re.match(r"^(?:[-*]|\d+[.)])\s+(.*)$", line)
        if bullet:
            items.append(bullet.group(1))
            continue
        flush()
        if not line:
            continue
        heading = re.match(r"^#{1,6}\s*(.*)$", line)
        out.append(f"<p><strong>{_e(heading.group(1))}</strong></p>" if heading else f"<p>{inline(line)}</p>")
    flush()
    return "".join(out)


def _fmt_duration(seconds, lang):
    seconds = int(seconds)
    if seconds >= 60:
        return f"{seconds // 60}:{seconds % 60:02d} {t('u_min', lang)}"
    return f"{seconds} {t('u_sec', lang)}"


def build_report(result, graph_page, *, name="document", lang="th", palette="lavender", dark=False, generated=None):
    """
    Standalone HTML page for an AnalysisResult. `graph_page` is the interactive graph document
    (render_graph_html output); it is embedded in a sandboxed iframe so the file works offline.
    """
    L = lambda key, **kw: t(key, lang, **kw)   # noqa: E731
    insights, relations, stats = result.insights, result.relations, result.stats
    kinds = type_names(lang)
    when = (generated or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")

    # ---- metrics ----
    cards = [
        (L("k_entities"), insights.get("total_entities", 0), ""),
        (L("k_relations"), insights.get("total_relations", 0), ""),
        (L("k_components"), insights.get("num_connected_components", 0), L("k_components_help")),
        (L("communities"), insights.get("num_communities", 0), ""),
        (L("density"), insights.get("density", 0), L("density_help")),
        (L("k_time"), _fmt_duration(stats.get("elapsed", 0), lang), ""),
    ]
    cards_html = "".join(
        f"<div class='card'><div class='k'>{_e(k)}</div><div class='v'>{_e(v)}</div>"
        + (f"<div class='hint'>{_e(h)}</div>" if h else "") + "</div>" for k, v, h in cards)

    # ---- legend ----
    swatches = node_colors(accent(palette, dark), "dark" if dark else "light")
    used = {"Topic"} | {r.get(k, "Entity") for r in relations for k in ("head_type", "tail_type")}
    legend = "".join(f"<span><i style='background:{swatches[k]['background']}'></i>{_e(kinds[k])}</span>"
                     for k in NODE_TYPES if k in used)

    # ---- insights ----
    hub = insights.get("top_entity_by_degree")
    bridge = insights.get("top_entity_by_betweenness")
    insight_cards = [
        f"<div class='card'><div class='k'>{_e(L('hub'))}</div><h3>{_e(hub['entity']) if hub else '—'}</h3>"
        f"<div class='hint'>{_e(L('hub_caption'))}</div></div>",
        (f"<div class='card'><div class='k'>{_e(L('bridge'))}</div><h3>{_e(bridge['entity'])}</h3>"
         f"<div class='hint'>{_e(L('bridge_caption'))}</div></div>") if bridge and bridge["score"] > 0 else
        f"<div class='card'><div class='k'>{_e(L('bridge'))}</div><div class='hint'>{_e(L('bridge_none'))}</div></div>",
    ]
    top_rel = insights.get("top_relations", [])
    peak = max((n for _, n in top_rel), default=1)
    bars = "".join(
        f"<div class='bar'><span>{_e(rel.replace('_', ' '))}</span>"
        f"<span class='track'><span class='fill' style='width:{100 * n / peak:.0f}%'></span></span>"
        f"<span class='n'>{n}</span></div>" for rel, n in top_rel)
    groups = "".join(
        f"<li><strong>{_e(L('group', n=i))}</strong> · {_e(' · '.join(members))}</li>"
        for i, members in enumerate(insights.get("sample_communities", []), 1))
    insights_html = (f"<div class='cols'>{''.join(insight_cards)}</div>"
                     + (f"<div class='card' style='margin-top:14px'><div class='k'>{_e(L('top_relations'))}</div>"
                        f"<div class='bars'>{bars}</div></div>" if bars else "")
                     + (f"<h3 style='margin:22px 0 10px;font-size:1rem;font-weight:500'>{_e(L('found_groups'))}</h3>"
                        f"<ul class='groups'>{groups}</ul>" if groups else ""))

    # ---- relations table ----
    rows = "".join(
        "<tr><td>{h}</td><td>{r}</td><td>{t}</td><td class='conf'>{c}</td><td class='src'>{s}</td></tr>".format(
            h=_e(r["head"]), r=_e(r["relation"]), t=_e(r["tail"]),
            c=f"{r.get('confidence', 0.0):.1f}", s=_e(r.get("source_sentence", "")))
        for r in relations)
    head = "".join(f"<th>{_e(L(k))}</th>" for k in ("c_subject", "c_relation", "c_object", "c_confidence", "c_source"))

    flagged = summary_flags(result) if result.summary else []
    summary = (highlight(summary_to_html(result.summary), flagged, L("ungrounded_title"))
               if result.summary else f"<p>{_e(L('summary_none'))}</p>")
    if flagged:
        shown = ", ".join(flagged[:10]) + (" …" if len(flagged) > 10 else "")
        summary = f"<div class='warn'>{_e(L('ungrounded_warn', n=len(flagged), terms=shown))}</div>" + summary
    elif result.summary and result.text:
        summary = f"<div class='ok'>{_e(L('grounded_ok'))}</div>" + summary
    truncated = (f"<span class='pill'>{_e(L('truncated_note', chars=stats.get('chars', 0)))}</span>"
                 if stats.get("truncated") else "")

    return f"""<!doctype html>
<html lang="{_e(lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(name)} · {_e(L('report_title'))}</title>
<style>@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap');
{css_vars(palette, dark)}
{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>{_e(L('report_title'))}</h1>
  <div class="meta"><span class="pill">{_e(name)}</span><span>{_e(L('report_generated', date=when))}</span>{truncated}</div>
</header>
<nav>
  <a href="#summary">{_e(L('tab_summary'))}</a><a href="#graph">{_e(L('tab_graph'))}</a>
  <a href="#insights">{_e(L('tab_insight'))}</a><a href="#relations">{_e(L('tab_table'))}</a>
</nav>
<div class="cards">{cards_html}</div>

<section id="summary"><h2>{_e(L('tab_summary'))}</h2>
  <div class="prose">{summary}</div><div class="note">{_e(L('summary_note'))}</div></section>

<section id="graph"><h2>{_e(L('tab_graph'))}</h2>
  <div class="legend">{legend}</div>
  <iframe class="graph" sandbox="allow-scripts" srcdoc="{_e(graph_page)}" title="{_e(L('tab_graph'))}"></iframe>
  <div class="note">{_e(L('graph_hint'))}</div></section>

<section id="insights"><h2>{_e(L('tab_insight'))}</h2>{insights_html}</section>

<section id="relations"><h2>{_e(L('tab_table'))}</h2>
  <div class="tools"><input id="filter" type="search" placeholder="{_e(L('search'))}" aria-label="{_e(L('search'))}">
    <span class="count" id="filter-count" data-template="{_e(L('report_count'))}"></span></div>
  <div class="tablebox"><table id="relations"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></section>

<footer>{_e(L('report_footer'))}</footer>
</div>
<script>{SCRIPT}</script>
</body>
</html>
"""
