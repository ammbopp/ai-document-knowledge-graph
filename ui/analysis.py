"""Document summary (size, estimate, warnings) and the analysis run with live progress."""
import re
import traceback

import streamlit as st

from src import pipeline

from .common import T, fmt_duration, get_embedder, get_nlp, text_hash


def render_document_summary(text, settings):
    """Metrics and warnings about the loaded document; returns its content hash."""
    words, chars = len(text.split()), len(text)

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric(T("words"), f"{words:,}")
    c2.metric(T("chars"), f"{chars:,}")
    c3.metric(T("est_time"), fmt_duration(pipeline.estimate_seconds(text[:settings.max_chars], settings.window_size),
                                          short=True), help=T("est_time_help"))

    if len(re.findall(r"[฀-๿]", text)) > 0.3 * max(chars, 1):
        st.warning(T("warn_thai"))
    if chars > settings.max_chars:
        st.info(T("info_truncate", chars=chars, limit=settings.max_chars))

    with st.expander(T("view_text")):
        st.text(text[:20000] + ("\n…" if chars > 20000 else ""))
    return text_hash(text)


def run_analysis(text, doc_name, settings, doc_hash):
    """Run the pipeline, store the result in session state and rerun (stops the script on failure)."""
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
            result = pipeline.analyze_document(
                text, nlp, embedder, min_confidence=settings.min_conf, threshold=settings.threshold,
                window_size=settings.window_size, max_chars=settings.max_chars, on_progress=on_progress,
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
    st.session_state.result, st.session_state.result_hash = result, doc_hash
    st.session_state.result_name = re.sub(r"[^\w\-]+", "_", doc_name) or "document"
    st.rerun()
