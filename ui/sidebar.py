"""Sidebar: language, Ollama status, colour theme, advanced analysis settings."""
from dataclasses import dataclass

import streamlit as st

from src.config import MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_SIZE
from src.i18n import DEFAULT_LANG, LANGS
from src.themes import PALETTES, accent

from .common import STRETCH, T, check_ollama, set_palette


@dataclass
class Settings:
    ollama_ok: bool
    min_conf: float
    threshold: float
    window_size: int
    max_chars: int


def render_sidebar(dark):
    with st.sidebar:
        st.markdown("<div class='brand'><span class='logo'>hub</span>Knowledge Graph</div>", unsafe_allow_html=True)
        st.segmented_control(T("lang"), list(LANGS), key="lang", format_func=LANGS.get,
                             label_visibility="collapsed", selection_mode="single")
        if st.session_state.lang not in LANGS:  # control was deselected
            st.session_state.lang = DEFAULT_LANG

        ollama_ok = _render_status()
        _render_theme(dark)

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
    return Settings(ollama_ok, min_conf, threshold, window_size, max_chars)


def _render_status():
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
    return ollama_ok


def _render_theme(dark):
    st.markdown(f"<div class='label'>{T('theme')}</div>", unsafe_allow_html=True)
    palette_ids = list(PALETTES)
    st.selectbox(T("theme"), [T("pal_" + k) for k in palette_ids], key="palette_choice",
                 index=palette_ids.index(st.session_state.palette), label_visibility="collapsed",
                 on_change=set_palette)
    st.markdown("<div class='swatches'>" + "".join(
        f"<span class='sw{' on' if name == st.session_state.palette else ''}' title='{T('pal_' + name)}' "
        f"style='background:{accent(name, dark)}'></span>" for name in PALETTES) + "</div>",
        unsafe_allow_html=True)
    st.caption(T("mode_hint"))
