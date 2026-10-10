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


def render_graph_html(G):
    tmp_dir = tempfile.mkdtemp(prefix="kg_")
    path = os.path.join(tmp_dir, "graph.html")
    try:
        visualize_graph(G, output_file=path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def analyze_document(text, nlp, embedder, *, min_confidence=MIN_CONFIDENCE, threshold=RESOLVER_THRESHOLD,
                     window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP, max_chars=None, on_progress=None):
    """
    Run the full pipeline. on_progress(percent, message, done) is called as work proceeds;
    done=True marks a finished step (message summarises its outcome).
    """
    def emit(pct, msg, done=False):
        if on_progress:
            on_progress(pct, msg, done)

    start = time.time()
    truncated = bool(max_chars) and len(text) > max_chars
    if truncated:
        text = truncate_at_sentence(text, max_chars)

    emit(2, "กำลังแก้คำสรรพนาม (Coreference Resolution)...")
    resolved_text = resolve_coreferences_with_llm(text)
    emit(8, "แก้คำสรรพนามเสร็จสิ้น", True)

    sentences = split_sentences(nlp, resolved_text)
    chunks = create_sliding_windows(sentences, window_size, overlap)
    emit(10, f"แบ่งเอกสารเป็น {len(sentences)} ประโยค / {len(chunks)} กลุ่ม", True)

    # Chunks are independent, so run several LLM requests at once; results keep document order
    per_chunk = [None] * len(chunks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, LLM_WORKERS)) as pool:
        futures = {pool.submit(extract_relations, [chunk]): i for i, chunk in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), 1):
            per_chunk[futures[future]] = future.result()
            remaining = (time.time() - t0) / done * (len(chunks) - done)
            emit(10 + int(70 * done / len(chunks)),
                 f"กำลังสกัดความสัมพันธ์ เสร็จแล้ว {done}/{len(chunks)} กลุ่ม (เหลือประมาณ {int(remaining)} วินาที)")
    relations = [r for chunk_relations in per_chunk for r in chunk_relations]
    usable = [r for r in relations if r.get("confidence", 0.0) >= min_confidence]
    emit(80, f"สกัดได้ {len(usable)} ความสัมพันธ์ที่ผ่านการตรวจสอบ", True)

    resolver = EntityResolver(threshold=threshold, model=embedder)
    resolved = resolve_relations(
        usable, resolver, lambda d, t: emit(80 + int(10 * d / t), f"กำลังรวมเอนทิตีที่ซ้ำกัน ({d}/{t})"))
    emit(90, f"หลังรวมเอนทิตีเหลือ {len(resolved)} ความสัมพันธ์", True)

    G = build_graph(resolved)
    html = render_graph_html(G)
    insights = analyze_graph(G)
    emit(95, f"สร้างกราฟ {G.number_of_nodes()} โหนด / {G.number_of_edges()} เส้น", True)

    emit(96, "กำลังสรุปเนื้อหา...")
    summary = generate_graph_augmented_summary(text, resolved) if resolved else ""
    emit(100, "เสร็จสิ้น", True)

    return AnalysisResult(resolved, insights, html, summary, {
        "elapsed": time.time() - start, "chunks": len(chunks), "extracted": len(relations),
        "usable": len(usable), "truncated": truncated, "chars": len(text),
    })
