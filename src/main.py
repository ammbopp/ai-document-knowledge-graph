import pathlib
import spacy
from fastcoref import FCoref

from .graph_insight import analyze_graph
from .entity_extraction import extract_entities
from .relation_extraction_ai import extract_relations
from .graph_builder import build_graph, visualize_graph
from .dependency_splitter import split_complex_sentence
from .entity_resolution import EntityResolver

DATA_PATH = pathlib.Path("data/sample_01.txt")
MIN_SENTENCE_LENGTH = 10

nlp = spacy.load("en_core_web_sm")

print("⏳ Loading Coreference Model (FCoref)...")
coref_model = FCoref(device='cpu') # เปลี่ยนเป็น 'cuda' ได้ถ้ามีการ์ดจอ

# =========================
# Helper Function: Sliding Window
# =========================
def create_sliding_windows(sentences, window_size=3, overlap=1):
    chunks = []
    step = max(1, window_size - overlap) 
    
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + window_size]
        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)
        
        if i + window_size >= len(sentences):
            break
            
    return chunks

# =========================
# 1. Load Document
# =========================
if not DATA_PATH.exists():
    print("⚠️ Sample file not found, creating dummy file...")
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w") as f:
        # ตัวอย่างประโยคที่มีการใช้สรรพนาม
        f.write("Elon Musk leads SpaceX. He also founded Tesla. It produces electric cars.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    original_text = f.read().strip()

print(f"📄 Loaded document: {DATA_PATH.name}")
print(f"   Original Text: {original_text[:100]}...")

# =========================
# 2. Coreference Resolution (Pre-processing)
# =========================
print("🔍 Running Coreference Resolution (Replacing pronouns)...")
# ให้โมเดลทำนายและแทนที่คำสรรพนามด้วยชื่อ Entity เต็ม
preds = coref_model.predict(texts=[original_text])
document_text = preds[0].get_resolved_text()

print(f"✅ Resolved Text: {document_text[:100]}...")

# =========================
# 3. Sentence Segmentation & Text Chunking
# =========================
# ใช้ Text ที่แก้สรรพนามแล้ว (document_text) มาตัดประโยค
doc = nlp(document_text)
sentences = [
    sent.text.strip()
    for sent in doc.sents
    if len(sent.text.strip()) >= MIN_SENTENCE_LENGTH
]

print(f"✂️ Initial sentences: {len(sentences)}")

# นำประโยคมามัดรวมแบบ Sliding Window (แนะนำ Window=3, Overlap=1)
chunks = create_sliding_windows(sentences, window_size=3, overlap=1)
print(f"📦 Grouped into {len(chunks)} chunks using sliding window.")

# =========================
# 4. Entity Extraction (Document-level)
# =========================
entities = extract_entities(document_text)

# =========================
# 5. Relation Extraction (LLM-based)
# =========================
# ส่ง Chunks ที่คลีนแล้วไปให้โมเดล Llama 3 สกัดความสัมพันธ์
relations = extract_relations(chunks)

print("\n=== RELATIONS (FILTERED) ===")
usable_relations = [
    r for r in relations
    if r.get("quality", "medium") != "low"
]

# =========================
# 5.5 Entity Resolution (ยุบรวม Node ที่ซ้ำซ้อน)
# =========================
print("\n=== RESOLVING ENTITIES (Similarity Threshold: 0.95) ===")
resolver = EntityResolver(threshold=0.95)
resolved_relations = []
seen_edges = set()

for r in usable_relations:
    # ส่ง type ไปด้วย โดยดึงจาก metadata ของความสัมพันธ์ (ถ้าไม่มีให้ใช้ "Entity")
    resolved_head = resolver.resolve(r["head"], r.get("head_type", "Entity"))
    resolved_tail = resolver.resolve(r["tail"], r.get("tail_type", "Entity"))
    
    edge_key = (resolved_head.lower(), r["relation"].lower(), resolved_tail.lower())
    if edge_key not in seen_edges:
        seen_edges.add(edge_key)
        
        new_r = r.copy()
        new_r["head"] = resolved_head
        new_r["tail"] = resolved_tail
        resolved_relations.append(new_r)

print("\n=== FINAL RELATIONS (AFTER RESOLUTION) ===")
for r in resolved_relations:
    print(f"{r['head']} -> {r['relation']} -> {r['tail']}")

# =========================
# 6. Knowledge Graph Construction (Mind Map Mode)
# =========================
topic_name = DATA_PATH.stem.replace("_", " ").title()

G = build_graph(resolved_relations, root_name=topic_name)

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