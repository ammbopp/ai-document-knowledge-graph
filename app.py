import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import pathlib

import streamlit as st

from src.themes import css_vars
from ui.analysis import render_document_summary, run_analysis
from ui.common import init_state, is_dark_theme
from ui.composer import render_composer, render_greeting, show_cards
from ui.results import render_results
from ui.sidebar import render_sidebar

st.set_page_config(page_title="Knowledge Graph", layout="wide", page_icon=":material/hub:",
                   initial_sidebar_state="expanded")
init_state()

DARK = is_dark_theme()
_css = pathlib.Path("assets/style.css")
if _css.exists():
    st.markdown(f"<style>{css_vars(st.session_state.palette, DARK)}\n{_css.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

settings = render_sidebar(DARK)
cards_slot = render_greeting()
composer = render_composer(settings.ollama_ok)

if not composer.text:
    show_cards(cards_slot)
    st.stop()

doc_hash = render_document_summary(composer.text, settings)
if composer.run:
    run_analysis(composer.text, composer.name, settings, doc_hash)
render_results(doc_hash, DARK)
