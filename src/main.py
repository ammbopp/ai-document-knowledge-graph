"""CLI entry point: python -m src.main [path/to/document.txt]"""
import argparse
import pathlib

import spacy

from .entity_resolution import load_embedder
from .pipeline import analyze_document

DEFAULT_PATH = pathlib.Path("data/sample_01.txt")
OUTPUT_PATH = pathlib.Path("output/knowledge_graph.html")


def _log(pct, msg, done):
    if done:
        print(f"[{pct:3d}%] {msg}")


def main(path=DEFAULT_PATH):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"❌ File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    print(f"📄 Loaded document: {path.name}")

    result = analyze_document(text, spacy.load("en_core_web_sm"), load_embedder(), on_progress=_log)

    print("\n=== FINAL RELATIONS ===")
    for r in result.relations:
        print(f"{r['head']} -> {r['relation']} -> {r['tail']}")
    print("\n=== GRAPH INSIGHTS ===")
    for k, v in result.insights.items():
        print(f"{k}: {v}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(result.html, encoding="utf-8")
    print(f"\n💾 Graph saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH)
    main(parser.parse_args().path)
