import networkx as nx
from collections import Counter

def analyze_graph(G: nx.DiGraph):
    """
    Thesis-level analysis of Knowledge Graph
    """
    insights = {}

    # =========================
    # 1. Filter low-quality edges
    # =========================
    Gf = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get("quality", "medium") != "low":
            Gf.add_edge(u, v, **data)

    insights["total_entities"] = Gf.number_of_nodes()
    insights["total_relations"] = Gf.number_of_edges()

    if Gf.number_of_nodes() == 0:
        return insights

    # =========================
    # 2. Centrality Analysis
    # =========================
    degree_centrality = nx.degree_centrality(Gf)
    betweenness = nx.betweenness_centrality(Gf)

    top_degree = max(degree_centrality, key=degree_centrality.get)
    top_bridge = max(betweenness, key=betweenness.get)

    insights["top_entity_by_degree"] = {
        "entity": top_degree,
        "score": round(degree_centrality[top_degree], 4)
    }

    insights["top_entity_by_betweenness"] = {
        "entity": top_bridge,
        "score": round(betweenness[top_bridge], 4)
    }

    # =========================
    # 3. Relation Distribution
    # =========================
    relations = []
    for _, _, data in Gf.edges(data=True):
        rel = data.get("relation") or data.get("relation_text")
        if rel:
            relations.append(rel)

    relation_count = Counter(relations)
    insights["top_relations"] = relation_count.most_common(5)

    # =========================
    # 4. Graph Quality Metrics
    # =========================
    undirected = Gf.to_undirected()

    insights["density"] = round(nx.density(Gf), 4)
    insights["num_connected_components"] = nx.number_connected_components(undirected)

    # =========================
    # 5. Community Detection (Theme Discovery)
    # =========================
    try:
        communities = list(nx.community.greedy_modularity_communities(undirected))
        insights["num_communities"] = len(communities)

        # แสดงเฉพาะ 3 community แรก (เขียนบท 4 ง่าย)
        insights["sample_communities"] = [
            list(c)[:5] for c in communities[:3]
        ]
    except Exception:
        insights["num_communities"] = 0
        insights["sample_communities"] = []

    return insights
