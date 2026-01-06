from .graph_insight import analyze_graph
from .entity_extraction import extract_entities
from .relation_extraction_ai import extract_relations
from .graph_builder import build_graph
from .graph_builder import visualize_graph
from .dependency_splitter import split_complex_sentence

with open("data/sample_02.txt", "r") as f:
    text = f.read()

sentences = [s.strip() for s in text.split(".") if s.strip()]

entities = extract_entities(text)
print("=== ENTITIES ===")
for e in entities:
    print("-", e)

expanded_sentences = []

for s in sentences:
    expanded_sentences.extend(split_complex_sentence(s))

relations = extract_relations(expanded_sentences)
print("\n=== RELATIONS (AI GENERATED) ===")
for r in relations:
    print(r)

usable_relations = [
    r for r in relations
    if r.get("quality") != "low"
]

G = build_graph(usable_relations)
insights = analyze_graph(G)

print("\n=== GRAPH INSIGHTS ===")
for k, v in insights.items():
    print(k, ":", v)

visualize_graph(G)
print("📊 Graph saved to graph.html")
