import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import hashlib
import inspect
import json
import pathlib
import re
import traceback

import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components

from src.config import MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_SIZE
from src.entity_resolution import load_embedder
from src.graph_view import NODE_TYPES, node_colors
from src.i18n import DEFAULT_LANG, LANGS, t
from src.llm import health
from src.pipeline import analyze_document, estimate_seconds, render_graph_html
from src.themes import DEFAULT_PALETTE, PALETTES, accent, accent_pair, css_vars
from src.web_fetch import FetchError, decode_upload, fetch_text_from_url

st.set_page_config(page_title="Knowledge Graph", layout="wide", page_icon=":material/hub:",
                   initial_sidebar_state="expanded")

# Streamlit renamed use_container_width -> width="stretch" and deprecated components.html -> st.iframe
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})

PASTE, UPLOAD, URL = "paste", "upload", "url"  # stable ids; labels come from i18n
SAMPLE_DIR = pathlib.Path("data")

for key, default in {
    "input_method": PASTE, "result": None, "result_hash": "", "result_name": "document",
    "url_text": "", "url_text_for": "", "url_error": "", "sample_name": "", "sample_text": "",
    "palette": DEFAULT_PALETTE, "lang": DEFAULT_LANG, "graph_cache": {},
}.items():
    st.session_state.setdefault(key, default)


def T(key, **kwargs):
    return t(key, st.session_state.lang, **kwargs)


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


def is_dark_theme():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark_theme()
_css = pathlib.Path("assets/style.css")
if _css.exists():
    st.markdown(f"<style>{css_vars(st.session_state.palette, DARK)}\n{_css.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ==========================================
# Cached resources
# ==========================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return load_embedder()


@st.cache_resource
def get_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_data(ttl=15, show_spinner=False)
def check_ollama(lang):
    return health(lang=lang)


def list_samples():
    samples = []
    for path in sorted(SAMPLE_DIR.glob("sample*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            samples.append((path.stem, text))
    return samples[:4]


def load_sample(name, text):
    st.session_state.input_method = PASTE
    st.session_state.paste_box = text
    st.session_state.sample_name, st.session_state.sample_text = name, text


def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fmt_duration(seconds, short=False):
    seconds = int(seconds)
    mins, secs = T("u_min"), T("u_sec")
    if short:  # compact form for metric values: 2:16 min
        return f"{seconds // 60}:{seconds % 60:02d} {mins}" if seconds >= 60 else f"{seconds} {secs}"
    return f"{seconds // 60} {mins} {seconds % 60} {secs}" if seconds >= 60 else f"{seconds} {secs}"


def set_palette():
    """Selectbox callback (runs before the rerun, so the new colours apply immediately)."""
    ids = list(PALETTES)
    labels = [T("pal_" + k) for k in ids]
    choice = st.session_state.get("palette_choice")
    if choice in labels:
        st.session_state.palette = ids[labels.index(choice)]


def graph_html(result):
    """Graph HTML in the current theme and language (re-rendered only when those change)."""
    if result.graph is None:
        return result.html
    cache_key = (st.session_state.palette, st.session_state.lang, DARK, id(result))
    cache = st.session_state.graph_cache
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = render_graph_html(result.graph, accent=accent_pair(st.session_state.palette),
                                             lang=st.session_state.lang, dark=DARK)
    return cache[cache_key]


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.markdown("<div class='brand'><span class='logo'>hub</span>Knowledge Graph</div>", unsafe_allow_html=True)
    st.segmented_control(T("lang"), list(LANGS), key="lang", format_func=LANGS.get,
                         label_visibility="collapsed", selection_mode="single")
    if st.session_state.lang not in LANGS:  # control was deselected
        st.session_state.lang = DEFAULT_LANG

    ollama_ok, ollama_msg = check_ollama(st.session_state.lang)
    st.markdown(f"<div class='label'>{T('status')}</div>", unsafe_allow_html=True)
    status_text, _, hint = ollama_msg.partition(" — ")
    st.markdown(f"<div class='status'><span class='dot {'ok' if ollama_ok else 'bad'}'></span>{status_text}</div>",
                unsafe_allow_html=True)
    if not ollama_ok:
        st.caption(hint)
        if st.button(T("recheck"), icon=":material/refresh:", **STRETCH):
            check_ollama.clear()
            st.rerun()

    st.markdown(f"<div class='label'>{T('theme')}</div>", unsafe_allow_html=True)
    palette_ids = list(PALETTES)
    st.selectbox(T("theme"), [T("pal_" + k) for k in palette_ids], key="palette_choice",
                 index=palette_ids.index(st.session_state.palette), label_visibility="collapsed",
                 on_change=set_palette)
    st.markdown("<div class='swatches'>" + "".join(
        f"<span class='sw{' on' if name == st.session_state.palette else ''}' title='{T('pal_' + name)}' "
        f"style='background:{accent(name, DARK)}'></span>" for name in PALETTES) + "</div>",
        unsafe_allow_html=True)
    st.caption(T("mode_hint"))

    st.markdown(f"<div class='label'>{T('analysis')}</div>", unsafe_allow_html=True)
    with st.expander(T("advanced")):
        min_conf = st.slider(T("min_conf"), 0.6, 1.0, float(MIN_CONFIDENCE), 0.1, help=T("min_conf_help"))
        threshold = st.slider(T("threshold"), 0.70, 0.99, float(RESOLVER_THRESHOLD), 0.01,
                              help=T("threshold_help"))
        window_size = st.slider(T("window"), 1, 5, WINDOW_SIZE, help=T("window_help"))
        max_chars = st.slider(T("max_chars"), 1000, 20000, 6000, 500, help=T("max_chars_help"))

    if st.session_state.result is not None and st.button(T("clear"), icon=":material/delete:", **STRETCH):
        st.session_state.result = None
        st.rerun()

    st.divider()
    st.caption(T("privacy"))

METHODS = {PASTE: T("m_paste"), UPLOAD: T("m_upload"), URL: T("m_url")}
NODE_LABELS = {k: T(k) for k in NODE_TYPES}

# ==========================================
# Greeting
# ==========================================
BUBBLE_ICON = (
    "<svg viewBox='0 0 36 36' fill='none'><g stroke='var(--accent)' stroke-width='1.6' opacity='.55'>"
    "<path d='M9 24 L18 9 L28 21 Z'/></g><g fill='var(--accent)'><circle cx='9' cy='24' r='4'/>"
    "<circle cx='18' cy='9' r='5'/><circle cx='28' cy='21' r='3.5'/></g></svg>")
st.markdown(
    f"""<div class="greet"><h1><span class="muted">{T('greet_muted')}</span> {T('greet_main')}</h1>
    <div class="bubble">{BUBBLE_ICON}<span>{T('bubble')}</span></div></div>""", unsafe_allow_html=True)

cards_slot = st.empty()  # feature cards, shown only while no document is loaded

# ==========================================
# Composer: input + method chips + analyse button
# ==========================================
method = st.session_state.input_method if st.session_state.input_method in METHODS else PASTE
main_text, doc_name = "", "document"
composer = st.container(border=True, key="composer")
with composer:
    if method == PASTE and not st.session_state.get("paste_box", "").strip():
        samples = list_samples()
        if samples:
            cols = st.columns([2.2] + [1] * len(samples) + [0.4], vertical_alignment="center")
            cols[0].markdown(f"<div class='composer-note'><span class='m'>auto_awesome</span>{T('try_sample')}</div>",
                             unsafe_allow_html=True)
            for col, (name, text) in zip(cols[1:], samples):
                col.button(name, key=f"sample_{name}", on_click=load_sample, args=(name, text), **STRETCH)

    if method == PASTE:
        main_text = st.text_area(T("text_label"), key="paste_box", height=190, label_visibility="collapsed",
                                 placeholder=T("text_placeholder")).strip()
        doc_name = st.session_state.sample_name if main_text == st.session_state.sample_text else "custom_text"
    elif method == UPLOAD:
        uploaded = st.file_uploader(T("file_label"), type=["txt"], label_visibility="collapsed")
        if uploaded is not None:
            main_text = decode_upload(uploaded.getvalue()).strip()
            doc_name = pathlib.Path(uploaded.name).stem
    else:
        col_url, col_btn = st.columns([6, 1.2], vertical_alignment="bottom")
        url = col_url.text_input(T("url_label"), placeholder="https://www.bbc.com/news/...",
                                 label_visibility="collapsed").strip()
        retry = col_btn.button(T("fetch"), icon=":material/sync:", **STRETCH)
        if url and (url != st.session_state.url_text_for or retry):
            with st.spinner(T("fetching")):
                try:
                    st.session_state.url_text = fetch_text_from_url(url).strip()
                    st.session_state.url_error = "" if st.session_state.url_text else "err_empty"
                except FetchError as e:
                    st.session_state.url_text, st.session_state.url_error = "", e.key
                except Exception as e:
                    st.session_state.url_text, st.session_state.url_error = "", str(e)
                st.session_state.url_text_for = url
        if url and url == st.session_state.url_text_for:
            if st.session_state.url_error:
                err = st.session_state.url_error
                st.error(T("fetch_failed", error=T(err) if err.startswith("err_") else err))
            else:
                main_text = st.session_state.url_text
                doc_name = "news_url"

    bar_left, bar_right = st.columns([5, 2], vertical_alignment="center")
    with bar_left:
        st.segmented_control(T("method_label"), list(METHODS), key="input_method", format_func=METHODS.get,
                             label_visibility="collapsed")
    with bar_right:
        run = st.button(T("run"), type="primary", icon=":material/arrow_upward:", icon_position="right",
                        disabled=not (main_text and ollama_ok), **STRETCH,
                        help=None if ollama_ok else T("run_help_no_ollama"))

if not main_text:
    cards_slot.markdown(
        f"""<div class="cards">
        <div class="fcard"><div class="ico">edit_note</div><p>{T('card1')}</p><small>{T('card1_tag')}</small></div>
        <div class="fcard"><div class="ico">account_tree</div><p>{T('card2')}</p><small>{T('card2_tag')}</small></div>
        <div class="fcard"><div class="ico">travel_explore</div><p>{T('card3')}</p><small>{T('card3_tag')}</small></div>
        </div>""", unsafe_allow_html=True)
    st.caption(T("empty_caption"))
    st.stop()

# ==========================================
# Document summary
# ==========================================
words, chars = len(main_text.split()), len(main_text)
cur_hash = text_hash(main_text)
will_truncate = chars > max_chars

st.write("")
c1, c2, c3 = st.columns(3)
c1.metric(T("words"), f"{words:,}")
c2.metric(T("chars"), f"{chars:,}")
c3.metric(T("est_time"), fmt_duration(estimate_seconds(main_text[:max_chars], window_size), short=True),
          help=T("est_time_help"))

if len(re.findall(r"[฀-๿]", main_text)) > 0.3 * max(chars, 1):
    st.warning(T("warn_thai"))
if will_truncate:
    st.info(T("info_truncate", chars=chars, limit=max_chars))

with st.expander(T("view_text")):
    st.text(main_text[:20000] + ("\n…" if chars > 20000 else ""))

if run:
    progress = st.progress(0, text=T("prep"))
    with st.status(T("analyzing"), expanded=True) as status:
        def on_progress(pct, msg, done):
            progress.progress(min(pct, 100), text=f"{msg} ({pct}%)")
            if done:
                st.write(f"{pct:>3}%  {msg}")
            else:
                status.update(label=msg)
        try:
            with st.spinner(T("loading_models")):
                embedder, nlp = get_embedder(), get_nlp()
            result = analyze_document(main_text, nlp, embedder, min_confidence=min_conf, threshold=threshold,
                                      window_size=window_size, max_chars=max_chars, on_progress=on_progress,
                                      lang=st.session_state.lang)
        except Exception as e:
            status.update(label=T("error_label"), state="error", expanded=False)
            failure = (e, traceback.format_exc())
        else:
            failure = None
            status.update(label=T("done_label", time=fmt_duration(result.stats["elapsed"])),
                          state="complete", expanded=False)
    if failure:  # shown outside the (collapsed) status box so it is always visible
        progress.empty()
        st.error(T("failed", error=failure[0]))
        with st.expander(T("error_details")):
            st.code(failure[1])
        st.stop()
    progress.empty()
    st.session_state.result, st.session_state.result_hash = result, cur_hash
    st.session_state.result_name = re.sub(r"[^\w\-]+", "_", doc_name) or "document"
    st.rerun()

# ==========================================
# Results
# ==========================================
result = st.session_state.result
if result is None:
    st.stop()
if st.session_state.result_hash != cur_hash:
    st.info(T("stale"))
    st.stop()

insights, relations, stats = result.insights, result.relations, result.stats
st.markdown(f"<div class='result-head'><h2>{T('results')}</h2><code>{st.session_state.result_name}</code></div>",
            unsafe_allow_html=True)

if not relations:
    st.warning(T("no_relations"))
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric(T("k_entities"), insights.get("total_entities", 0))
k2.metric(T("k_relations"), insights.get("total_relations", 0))
k3.metric(T("k_components"), insights.get("num_connected_components", 0), help=T("k_components_help"))
k4.metric(T("k_time"), fmt_duration(stats["elapsed"], short=True))
if stats["truncated"]:
    st.caption(T("truncated_note", chars=stats["chars"]))

st.write("")
tab_graph, tab_table, tab_insight, tab_summary = st.tabs(
    [T("tab_graph"), T("tab_table"), T("tab_insight"), T("tab_summary")])

with tab_graph:
    swatches = node_colors(accent(st.session_state.palette, DARK), "dark" if DARK else "light")
    used = {"Topic"} | {r.get(k, "Entity") for r in relations for k in ("head_type", "tail_type")}
    legend = "".join(
        f"<span><i style='background:{swatches[kind]['background']}'></i>{NODE_LABELS[kind]}</span>"
        for kind in NODE_TYPES if kind in used)
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    st.caption(T("graph_hint"))
    html_graph = graph_html(result)
    show_html(html_graph, 680)
    st.download_button(T("dl_graph"), html_graph, icon=":material/download:",
                       file_name=f"{st.session_state.result_name}_graph.html", mime="text/html")

with tab_table:
    C_SUBJECT, C_REL, C_OBJECT = T("c_subject"), T("c_relation"), T("c_object")
    C_CONF, C_SOURCE = T("c_confidence"), T("c_source")
    df = pd.DataFrame([{
        C_SUBJECT: r["head"], C_REL: r["relation"], C_OBJECT: r["tail"],
        T("c_subject_type"): NODE_LABELS.get(r.get("head_type", "Entity"), NODE_LABELS["Entity"]),
        T("c_object_type"): NODE_LABELS.get(r.get("tail_type", "Entity"), NODE_LABELS["Entity"]),
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
                       file_name=f"{st.session_state.result_name}_relations.csv", mime="text/csv", **STRETCH)
    d2.download_button(T("dl_json"), json.dumps(relations, ensure_ascii=False, indent=2),
                       icon=":material/download:", file_name=f"{st.session_state.result_name}_relations.json",
                       mime="application/json", **STRETCH)

with tab_insight:
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
                     color=accent(st.session_state.palette, DARK))

    communities = insights.get("sample_communities", [])
    if communities:
        st.markdown(f"<div class='label'>{T('found_groups')}</div>", unsafe_allow_html=True)
        for i, members in enumerate(communities, 1):
            st.markdown(f"**{T('group', n=i)}** · " + " · ".join(members))
    with st.expander(T("raw_json")):
        st.json(insights)

with tab_summary:
    # LLMs often emit "• a • b" on one line; turn inline bullets into a real Markdown list
    summary_md = re.sub(r"\s*•\s*", "\n\n- ", result.summary) if result.summary else ""
    st.markdown(summary_md or T("summary_none"))
    st.caption(T("summary_note"))
    if result.summary:
        st.download_button(T("dl_summary"), result.summary, icon=":material/download:",
                           file_name=f"{st.session_state.result_name}_summary.md", mime="text/markdown")
