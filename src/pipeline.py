"""End-to-end document analysis, independent of any UI."""
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .config import LLM_WORKERS, MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_OVERLAP, WINDOW_SIZE
from .entity_resolution import EntityResolver, resolve_relations
from .graph_builder import build_graph, visualize_graph
from .graph_insight import analyze_graph
from .i18n import graph_labels, t, type_names
from .preprocessing import resolve_coreferences_with_llm
from .relation_extraction_ai import extract_relations
from .summarization import generate_graph_augmented_summary
from .text_utils import create_sliding_windows, split_sentences

SECONDS_PER_CHUNK = 25  # effective seconds per chunk with parallel llama3 requests (estimates only)


@dataclass
class AnalysisResult:
    relations: list
    insights: dict
    html: str
    summary: str
    graph: object = None
    stats: dict = field(default_factory=dict)


def truncate_at_sentence(text, max_chars):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > max_chars * 0.5 else cut


def estimate_seconds(text, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    sentences = max(1, len(text) // 120)  # ~120 chars per sentence
    step = max(1, window_size - overlap)
    return int(max(1, -(-sentences // step)) * SECONDS_PER_CHUNK)


def extract_chunks(chunks, on_chunk=None):
    """
    Extract relations from every chunk with several LLM requests in flight; results keep document
    order. on_chunk(done, total, remaining_seconds) is called as each chunk completes.
    """
    per_chunk = [None] * len(chunks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, LLM_WORKERS)) as pool:
        futures = {pool.submit(extract_relations, [chunk]): i for i, chunk in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), 1):
            per_chunk[futures[future]] = future.result()
            if on_chunk:
                on_chunk(done, len(chunks), int((time.time() - t0) / done * (len(chunks) - done)))
    return per_chunk


def render_graph_html(G, *, accent=("#6b63d9", "#a9a3ff"), lang="th", dark=False):
    """Render the interactive graph with theme colours and UI strings for `lang`."""
    tmp_dir = tempfile.mkdtemp(prefix="kg_")
    path = os.path.join(tmp_dir, "graph.html")
    try:
        visualize_graph(G, output_file=path, accent=accent, labels=graph_labels(lang),
                        type_names=type_names(lang), dark=dark)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def analyze_document(text, nlp, embedder, *, min_confidence=MIN_CONFIDENCE, threshold=RESOLVER_THRESHOLD,
                     window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP, max_chars=None, on_progress=None,
                     lang="th"):
    """
    Run the full pipeline. on_progress(percent, message, done) is called as work proceeds;
    done=True marks a finished step (message summarises its outcome). Messages use `lang`.
    """
    def emit(pct, key, finished=False, **kwargs):
        if on_progress:
            on_progress(pct, t(key, lang, **kwargs), finished)

    start = time.time()
    truncated = bool(max_chars) and len(text) > max_chars
    if truncated:
        text = truncate_at_sentence(text, max_chars)

    marks = {"start": start}  # stage boundaries -> per-stage seconds in stats

    emit(2, "p_coref")
    resolved_text = resolve_coreferences_with_llm(text)
    marks["coref"] = time.time()
    emit(8, "p_coref_done", True)

    sentences = split_sentences(nlp, resolved_text)
    chunks = create_sliding_windows(sentences, window_size, overlap)
    marks["split"] = time.time()
    emit(10, "p_split", True, sentences=len(sentences), chunks=len(chunks))

    def on_chunk(done, total, remaining):
        emit(10 + int(70 * done / total), "p_extract", done=done, total=total, remaining=remaining)

    relations = [r for chunk_relations in extract_chunks(chunks, on_chunk) for r in chunk_relations]
    usable = [r for r in relations if r.get("confidence", 0.0) >= min_confidence]
    marks["extract"] = time.time()
    emit(80, "p_extracted", True, count=len(usable))

    resolver = EntityResolver(threshold=threshold, model=embedder)
    resolved = resolve_relations(
        usable, resolver, lambda d, n: emit(80 + int(10 * d / n), "p_merge", done=d, total=n))
    marks["resolve"] = time.time()
    emit(90, "p_merged", True, count=len(resolved))

    G = build_graph(resolved)
    html = render_graph_html(G, lang=lang)
    insights = analyze_graph(G)
    marks["graph"] = time.time()
    emit(95, "p_graph", True, nodes=G.number_of_nodes(), edges=G.number_of_edges())

    emit(96, "p_summary")
    summary = generate_graph_augmented_summary(text, resolved, lang) if resolved else ""
    marks["summary"] = time.time()
    emit(100, "p_done", True)

    names = list(marks)
    stage_seconds = {cur: marks[cur] - marks[prev] for prev, cur in zip(names, names[1:])}
    return AnalysisResult(resolved, insights, html, summary, G, {
        "elapsed": time.time() - start, "chunks": len(chunks), "sentences": len(sentences),
        "extracted": len(relations), "usable": len(usable), "truncated": truncated,
        "chars": len(text), "stage_seconds": stage_seconds,
    })
