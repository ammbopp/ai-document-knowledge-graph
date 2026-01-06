import networkx as nx
from pyvis.network import Network

def build_graph(relations):
    G = nx.DiGraph()

    for r in relations:
        head = r["head"]
        tail = r["tail"]
        rel_text = r["relation_text"]   # ✅ ใช้ relation_text

        G.add_edge(
            head,
            tail,
            relation_text=rel_text,
            sentence=r.get("source_sentence", "")
        )

    return G


def visualize_graph(G, output_file="graph.html"):
    net = Network(height="750px", width="100%", directed=True)

    for node in G.nodes():
        net.add_node(node, label=node)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("relation_text", ""))

    net.write_html(output_file)
    print(f"📊 Graph saved to {output_file}")