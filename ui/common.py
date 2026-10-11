"""Shared Streamlit helpers: session state, translation, cached resources and small formatters."""
import hashlib
import inspect

import spacy
import streamlit as st
import streamlit.components.v1 as components

from src import entity_resolution
from src.i18n import DEFAULT_LANG, t
from src.graph_view import NODE_TYPES
from src.llm import health
from src.pipeline import render_graph_html
from src.themes import DEFAULT_PALETTE, PALETTES, accent_pair

# Streamlit renamed use_container_width -> width="stretch" and deprecated components.html -> st.iframe
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})

PASTE, UPLOAD, URL = "paste", "upload", "url"  # stable ids; labels come from i18n


def init_state():
    for key, default in {
        "input_method": PASTE, "result": None, "result_hash": "", "result_name": "document",
        "url_text": "", "url_text_for": "", "url_error": "", "sample_name": "", "sample_text": "",
        "palette": DEFAULT_PALETTE, "lang": DEFAULT_LANG, "graph_cache": {},
    }.items():
        st.session_state.setdefault(key, default)


def T(key, **kwargs):
    return t(key, st.session_state.lang, **kwargs)


def node_labels():
    return {k: T(k) for k in NODE_TYPES}


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


def graph_html(result, dark):
    """Graph HTML in the current theme and language (re-rendered only when those change)."""
    if result.graph is None:
        return result.html
    cache_key = (st.session_state.palette, st.session_state.lang, dark, id(result))
    cache = st.session_state.graph_cache
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = render_graph_html(result.graph, accent=accent_pair(st.session_state.palette),
                                             lang=st.session_state.lang, dark=dark)
    return cache[cache_key]


# ---------- cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return entity_resolution.load_embedder()


@st.cache_resource
def get_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_data(ttl=15, show_spinner=False)
def check_ollama(lang):
    return health(lang=lang)
