"""Greeting, feature cards and the input composer (paste / upload / URL + analyse button)."""
import pathlib
from dataclasses import dataclass

import streamlit as st

from src.web_fetch import FetchError, decode_upload, fetch_text_from_url

from .common import PASTE, STRETCH, UPLOAD, URL, T

SAMPLE_DIR = pathlib.Path("data")

BUBBLE_ICON = (
    "<svg viewBox='0 0 36 36' fill='none'><g stroke='var(--accent)' stroke-width='1.6' opacity='.55'>"
    "<path d='M9 24 L18 9 L28 21 Z'/></g><g fill='var(--accent)'><circle cx='9' cy='24' r='4'/>"
    "<circle cx='18' cy='9' r='5'/><circle cx='28' cy='21' r='3.5'/></g></svg>")


@dataclass
class Composer:
    text: str
    name: str
    run: bool


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


def render_greeting():
    """Headline + hint bubble; returns the slot where the feature cards appear."""
    st.markdown(
        f"""<div class="greet"><h1><span class="muted">{T('greet_muted')}</span> {T('greet_main')}</h1>
        <div class="bubble">{BUBBLE_ICON}<span>{T('bubble')}</span></div></div>""", unsafe_allow_html=True)
    return st.empty()


def show_cards(slot):
    """Feature cards, shown only while no document is loaded."""
    slot.markdown(
        f"""<div class="cards">
        <div class="fcard"><div class="ico">edit_note</div><p>{T('card1')}</p><small>{T('card1_tag')}</small></div>
        <div class="fcard"><div class="ico">account_tree</div><p>{T('card2')}</p><small>{T('card2_tag')}</small></div>
        <div class="fcard"><div class="ico">travel_explore</div><p>{T('card3')}</p><small>{T('card3_tag')}</small></div>
        </div>""", unsafe_allow_html=True)
    st.caption(T("empty_caption"))


def render_composer(ollama_ok):
    methods = {PASTE: T("m_paste"), UPLOAD: T("m_upload"), URL: T("m_url")}
    method = st.session_state.input_method if st.session_state.input_method in methods else PASTE
    text, name = "", "document"
    with st.container(border=True, key="composer"):
        if method == PASTE:
            _sample_chips()
            text, name = _paste_input()
        elif method == UPLOAD:
            text, name = _upload_input()
        else:
            text, name = _url_input()

        bar_left, bar_right = st.columns([5, 2], vertical_alignment="center")
        with bar_left:
            st.segmented_control(T("method_label"), list(methods), key="input_method", format_func=methods.get,
                                 label_visibility="collapsed")
        with bar_right:
            run = st.button(T("run"), type="primary", icon=":material/arrow_upward:", icon_position="right",
                            disabled=not (text and ollama_ok), **STRETCH,
                            help=None if ollama_ok else T("run_help_no_ollama"))
    return Composer(text, name, run)


def _sample_chips():
    if st.session_state.get("paste_box", "").strip():
        return
    samples = list_samples()
    if not samples:
        return
    cols = st.columns([2.2] + [1] * len(samples) + [0.4], vertical_alignment="center")
    cols[0].markdown(f"<div class='composer-note'><span class='m'>auto_awesome</span>{T('try_sample')}</div>",
                     unsafe_allow_html=True)
    for col, (sample_name, sample_text) in zip(cols[1:], samples):
        col.button(sample_name, key=f"sample_{sample_name}", on_click=load_sample, args=(sample_name, sample_text),
                   **STRETCH)


def _paste_input():
    text = st.text_area(T("text_label"), key="paste_box", height=190, label_visibility="collapsed",
                        placeholder=T("text_placeholder")).strip()
    name = st.session_state.sample_name if text == st.session_state.sample_text else "custom_text"
    return text, name


def _upload_input():
    uploaded = st.file_uploader(T("file_label"), type=["txt"], label_visibility="collapsed")
    if uploaded is None:
        return "", "document"
    return decode_upload(uploaded.getvalue()).strip(), pathlib.Path(uploaded.name).stem


def _url_input():
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
            return st.session_state.url_text, "news_url"
    return "", "document"
