"""Results area: key metrics, the full-report download and the graph / relations / insights / summary tabs."""
import json

import pandas as pd
import streamlit as st

from src.graph_view import NODE_TYPES, node_colors
from src.grounding import highlight, summary_flags
from src.report import build_report, summary_to_html
from src.themes import accent

from .common import STRETCH, T, fmt_duration, graph_html, node_labels, show_html


def render_results(doc_hash, dark):
    result = st.session_state.result
    if result is None:
        st.stop()
    if st.session_state.result_hash != doc_hash:
        st.info(T("stale"))
        st.stop()

    insights, relations, stats = result.insights, result.relations, result.stats
    name = st.session_state.result_name

    head_col, report_col = st.columns([3, 1.3], vertical_alignment="bottom")
    head_col.markdown(f"<div class='result-head'><h2>{T('results')}</h2><code>{name}</code></div>",
                      unsafe_allow_html=True)

    if not relations:
        st.warning(T("no_relations"))
        st.stop()

    report_col.download_button(
        T("dl_report"),
        build_report(result, graph_html(result, dark), name=name, lang=st.session_state.lang,
                     palette=st.session_state.palette, dark=dark),
        icon=":material/download:", file_name=f"{name}_report.html", mime="text/html", type="primary", **STRETCH)

    _render_metrics(insights, stats)

    st.write("")
    tab_graph, tab_table, tab_insight, tab_summary = st.tabs(
        [T("tab_graph"), T("tab_table"), T("tab_insight"), T("tab_summary")])
    with tab_graph:
        _graph_tab(result, relations, name, dark)
    with tab_table:
        _table_tab(relations, name)
    with tab_insight:
        _insight_tab(insights, dark)
    with tab_summary:
        _summary_tab(result, name)


def _render_metrics(insights, stats):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(T("k_entities"), insights.get("total_entities", 0))
    k2.metric(T("k_relations"), insights.get("total_relations", 0))
    k3.metric(T("k_components"), insights.get("num_connected_components", 0), help=T("k_components_help"))
    k4.metric(T("k_time"), fmt_duration(stats["elapsed"], short=True))
    if stats["truncated"]:
        st.caption(T("truncated_note", chars=stats["chars"]))


def _graph_tab(result, relations, name, dark):
    labels = node_labels()
    swatches = node_colors(accent(st.session_state.palette, dark), "dark" if dark else "light")
    used = {"Topic"} | {r.get(k, "Entity") for r in relations for k in ("head_type", "tail_type")}
    legend = "".join(
        f"<span><i style='background:{swatches[kind]['background']}'></i>{labels[kind]}</span>"
        for kind in NODE_TYPES if kind in used)
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    st.caption(T("graph_hint"))
    html_graph = graph_html(result, dark)
    show_html(html_graph, 680)
    st.download_button(T("dl_graph"), html_graph, icon=":material/download:",
                       file_name=f"{name}_graph.html", mime="text/html")


def _table_tab(relations, name):
    labels = node_labels()
    C_SUBJECT, C_REL, C_OBJECT = T("c_subject"), T("c_relation"), T("c_object")
    C_CONF, C_SOURCE = T("c_confidence"), T("c_source")
    df = pd.DataFrame([{
        C_SUBJECT: r["head"], C_REL: r["relation"], C_OBJECT: r["tail"],
        T("c_subject_type"): labels.get(r.get("head_type", "Entity"), labels["Entity"]),
        T("c_object_type"): labels.get(r.get("tail_type", "Entity"), labels["Entity"]),
        C_CONF: r.get("confidence", 0.0), C_SOURCE: r.get("source_sentence", ""),
    } for r in relations])

    query = st.text_input(T("search"), placeholder=T("search"), label_visibility="collapsed").strip().lower()
    if query:
        df = df[df.apply(lambda row: query in " ".join(map(str, row.values)).lower(), axis=1)]
    st.caption(T("showing", shown=len(df), total=len(relations)))

    event = st.dataframe(
        df, **STRETCH, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_order=[C_SUBJECT, C_REL, C_OBJECT, C_CONF, C_SOURCE],
        column_config={
            C_CONF: st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.1f"),
            C_SOURCE: st.column_config.TextColumn(width="large"),
        })
    if event.selection.rows and event.selection.rows[0] < len(df):
        row = df.iloc[event.selection.rows[0]]
        with st.container(border=True):
            st.markdown(f"**{row[C_SUBJECT]}** → *{row[C_REL]}* → **{row[C_OBJECT]}**")
            st.caption(f"“{row[C_SOURCE]}”")

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button(T("dl_csv"), df.to_csv(index=False).encode("utf-8-sig"), icon=":material/download:",
                       file_name=f"{name}_relations.csv", mime="text/csv", **STRETCH)
    d2.download_button(T("dl_json"), json.dumps(relations, ensure_ascii=False, indent=2),
                       icon=":material/download:", file_name=f"{name}_relations.json",
                       mime="application/json", **STRETCH)


def _insight_tab(insights, dark):
    i1, i2, i3 = st.columns(3)
    i1.metric(T("density"), insights.get("density", 0), help=T("density_help"))
    i2.metric(T("communities"), insights.get("num_communities", 0))
    i3.metric(T("k_components"), insights.get("num_connected_components", 0))

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown(f"<div class='label'>{T('hub')}</div>", unsafe_allow_html=True)
        top = insights.get("top_entity_by_degree")
        if top:
            st.markdown(f"### {top['entity']}")
            st.caption(T("hub_caption"))
    with right:
        st.markdown(f"<div class='label'>{T('bridge')}</div>", unsafe_allow_html=True)
        bridge = insights.get("top_entity_by_betweenness")
        if bridge and bridge["score"] > 0:
            st.markdown(f"### {bridge['entity']}")
            st.caption(T("bridge_caption"))
        else:
            st.caption(T("bridge_none"))

    top_rel = insights.get("top_relations", [])
    if top_rel:
        st.markdown(f"<div class='label'>{T('top_relations')}</div>", unsafe_allow_html=True)
        rel_df = pd.DataFrame([(name.replace("_", " "), n) for name, n in top_rel],
                              columns=[T("c_relation"), T("count")]).set_index(T("c_relation"))
        st.bar_chart(rel_df, horizontal=True, height=40 * len(rel_df) + 60,
                     color=accent(st.session_state.palette, dark))

    communities = insights.get("sample_communities", [])
    if communities:
        st.markdown(f"<div class='label'>{T('found_groups')}</div>", unsafe_allow_html=True)
        for i, members in enumerate(communities, 1):
            st.markdown(f"**{T('group', n=i)}** · " + " · ".join(members))
    with st.expander(T("raw_json")):
        st.json(insights)


def _summary_tab(result, name):
    if not result.summary:
        st.markdown(T("summary_none"))
        st.caption(T("summary_note"))
        return
    flagged = summary_flags(result)
    if flagged:
        st.warning(T("ungrounded_warn", n=len(flagged), terms=", ".join(flagged[:10]) + (" …" if len(flagged) > 10 else "")),
                   icon=":material/fact_check:")
    elif result.text:
        st.caption(":material/check_circle: " + T("grounded_ok"))
    # summary_to_html turns LLM bullets into a list and escapes everything before the marks are added
    st.markdown(highlight(summary_to_html(result.summary), flagged, T("ungrounded_title")), unsafe_allow_html=True)
    st.caption(T("summary_note"))
    if result.summary:
        st.download_button(T("dl_summary"), result.summary, icon=":material/download:",
                           file_name=f"{name}_summary.md", mime="text/markdown")
