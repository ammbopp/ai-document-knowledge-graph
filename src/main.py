"""CLI entry point: python -m src.main [path/to/document.txt]"""
import argparse
import pathlib

import spacy

from .config import MIN_CONFIDENCE, RESOLVER_THRESHOLD
from .entity_resolution import EntityResolver, resolve_relations
from .graph_builder import build_graph, visualize_graph
from .graph_insight import analyze_graph
from .preprocessing import resolve_coreferences_with_llm
from .relation_extraction_ai import extract_relations
from .text_utils import create_sliding_windows, split_sentences

DEFAULT_PATH = pathlib.Path("data/sample_01.txt")
OUTPUT_PATH = pathlib.Path("output/knowledge_graph.html")


def main(path=DEFAULT_PATH):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"❌ File not found: {path}")

    original_text = path.read_text(encoding="utf-8").strip()
    print(f"📄 Loaded document: {path.name}")

    print("🔍 Running coreference resolution (LLM)...")
    document_text = resolve_coreferences_with_llm(original_text)

    nlp = spacy.load("en_core_web_sm")
    sentences = split_sentences(nlp, document_text)
    chunks = create_sliding_windows(sentences)
    print(f"✂️ {len(sentences)} sentences -> {len(chunks)} chunks")

    relations = extract_relations(chunks)
    usable = [r for r in relations if r.get("confidence", 0.0) >= MIN_CONFIDENCE]

    resolver = EntityResolver(threshold=RESOLVER_THRESHOLD)
    resolved = resolve_relations(usable, resolver)

    print("\n=== FINAL RELATIONS (AFTER RESOLUTION) ===")
    for r in resolved:
        print(f"{r['head']} -> {r['relation']} -> {r['tail']}")

    G = build_graph(resolved)
    print(f"\n🧠 Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    print("\n=== GRAPH INSIGHTS ===")
    for k, v in analyze_graph(G).items():
        print(f"{k}: {v}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    visualize_graph(G, output_file=str(OUTPUT_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH)
    main(parser.parse_args().path)
