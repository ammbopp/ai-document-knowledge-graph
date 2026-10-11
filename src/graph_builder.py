import networkx as nx
from .graph_view import NODE_TYPES, node_colors, visualize_graph  # noqa: F401  (re-exported)

# =========================
# Utilities
# =========================
def normalize_node(text: str) -> str:
    if not text:
        return "Unknown"
    return " ".join(text.strip().split())

def normalize_relation(rel: str) -> str:
    return rel.lower().strip().replace(" ", "_")

# =========================
# Graph Construction (Mind Map Logic)
# =========================

def build_graph(relations):
    """
    Build a knowledge graph (MultiDiGraph: several relations / both directions may link a pair)
    and connect disconnected clusters to the Main Entity with virtual edges (type="virtual").
    """
    G = nx.MultiDiGraph()

    # 1. Build the graph from all relations
    for r in relations:
        head = normalize_node(r["head"])
        tail = normalize_node(r["tail"])
        relation = normalize_relation(
            r.get("relation", r.get("relation_text", "related_to"))
        )
        sentence = r.get("source_sentence", "")

        for node, node_type in ((head, r.get("head_type", "Entity")), (tail, r.get("tail_type", "Entity"))):
            if not G.has_node(node):
                G.add_node(node, type=node_type, confidence=r.get("confidence", 1.0))
            elif G.nodes[node].get("type", "Entity") == "Entity" and node_type != "Entity":
                G.nodes[node]["type"] = node_type

        if G.has_edge(head, tail, key=relation):
            edge = G[head][tail][relation]
            edge["count"] += 1
            edge["sentences"].append(sentence)
        else:
            G.add_edge(
                head, tail, key=relation,
                relation=relation,
                count=1,
                sentences=[sentence],
                confidence=r.get("confidence", 1.0),
            )

    if G.number_of_nodes() == 0:
        return G

    # 2. Main Entity = highest degree (ties broken alphabetically so the result is deterministic)
    degrees = dict(G.degree())
    main_entity = min(degrees, key=lambda n: (-degrees[n], n))
    print(f"🌟 AI Identified Main Entity: {main_entity}")

    G.nodes[main_entity]["type"] = "Topic"
    G.nodes[main_entity]["group"] = "Hub"

    # 3. Link each disconnected cluster's best node to the Main Entity
    components = sorted(nx.connected_components(G.to_undirected()), key=lambda c: min(c))
    for component in components:
        if main_entity in component:
            continue
        sub_degrees = {n: degrees[n] for n in component}
        representative = min(sub_degrees, key=lambda n: (-sub_degrees[n], n))
        G.add_edge(
            main_entity, representative,
            key="related_to",
            relation="related_to",
            count=1,
            sentences=["(Implicit connection to main entity)"],
            confidence=1.0,
            type="virtual",
        )

    return G
