import networkx as nx
from collections import Counter

def analyze_graph(G: nx.DiGraph):
    """
    วิเคราะห์ insight จาก Knowledge Graph
    """
    insights = {}
    # 🔥 สร้างกราฟเฉพาะ relation ที่ไม่ใช่ noise
    G_filtered = nx.DiGraph()

    for u, v, data in G.edges(data=True):
        if data.get("quality") != "low":
            G_filtered.add_edge(u, v, **data)

    # 1. หา entity ที่สำคัญที่สุด (hub)
    degree_centrality = nx.degree_centrality(G_filtered)
    if degree_centrality:
        top_node = max(degree_centrality, key=degree_centrality.get)
        insights["top_node"] = top_node
        insights["degree_centrality"] = degree_centrality[top_node]
    else:
        insights["top_node"] = None
        insights["degree_centrality"] = 0

    # 2. ความสัมพันธ์ที่พบบ่อยที่สุด
    relations = []
    for _, _, data in G.edges(data=True):
        # 🔥 กรอง relation ที่ quality ต่ำ
        if data.get("quality") == "low":
            continue

        if "relation_text" in data:
            relations.append(data["relation_text"])

    if relations:
        relation_count = Counter(relations)
        insights["top_relation"] = relation_count.most_common(1)[0]
    else:
        insights["top_relation"] = None

    # 3. สถิติโดยรวม
    insights["total_entities"] = G_filtered.number_of_nodes()
    insights["total_relations"] = G_filtered.number_of_edges()

    return insights
