import pathlib
import spacy

from .graph_insight import analyze_graph
from .entity_extraction import extract_entities
from .relation_extraction_ai import extract_relations
from .graph_builder import build_graph, visualize_graph
from .dependency_splitter import split_complex_sentence
from .entity_resolution import EntityResolver

DATA_PATH = pathlib.Path("data/sample_03.txt")
MIN_SENTENCE_LENGTH = 10

nlp = spacy.load("en_core_web_sm")

# =========================
# 1. Load Document
# =========================
if not DATA_PATH.exists():
    print("⚠️ Sample file not found, creating dummy file...")
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w") as f:
        f.write("Google works with OpenAI. Microsoft develops Azure. Elon Musk leads SpaceX.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    document_text = f.read().strip()

print(f"📄 Loaded document: {DATA_PATH.name}")

# =========================
# 2. Sentence Segmentation
# =========================
doc = nlp(document_text)
sentences = [
    sent.text.strip()
    for sent in doc.sents
    if len(sent.text.strip()) >= MIN_SENTENCE_LENGTH
]

print(f"✂️ Initial sentences: {len(sentences)}")

# =========================
# 3. Complex Sentence Decomposition
# =========================

expanded_sentences = sentences
""" expanded_sentences = []
for s in sentences:
    expanded_sentences.extend(split_complex_sentence(s)) """

print(f"🔍 Expanded sentences: {len(expanded_sentences)}")

# =========================
# 4. Entity Extraction (Document-level)
# =========================
entities = extract_entities(document_text)

# =========================
# 5. Relation Extraction (LLM-based)
# =========================
relations = extract_relations(expanded_sentences)

print("\n=== RELATIONS (FILTERED) ===")
usable_relations = [
    r for r in relations
    if r.get("quality", "medium") != "low"
]

for r in usable_relations:
    print(f"{r['head']} -> {r['relation']} -> {r['tail']}")

# =========================
# 6. Knowledge Graph Construction (Mind Map Mode)
# =========================
# Use filename as main topic
topic_name = DATA_PATH.stem.replace("_", " ").title()

G = build_graph(usable_relations, root_name=topic_name)

print(f"\n🧠 Knowledge Graph constructed (Mind Map Style)")
print(f"   • Root Node: {topic_name}")
print(f"   • Total Nodes: {G.number_of_nodes()}")
print(f"   • Total Edges: {G.number_of_edges()}")

# =========================
# 7. Graph Analysis
# =========================
insights = analyze_graph(G)
print("\n=== GRAPH INSIGHTS ===")
for k, v in insights.items():
    print(f"{k}: {v}")

# =========================
# 8. Visualization
# =========================
OUTPUT_PATH = "output/knowledge_graph.html"
pathlib.Path("output").mkdir(exist_ok=True)

visualize_graph(G, output_file=OUTPUT_PATH)