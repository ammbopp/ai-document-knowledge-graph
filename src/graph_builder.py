import networkx as nx
from pyvis.network import Network

def clean_node(text):
    if not text:
        return "Unknown"
    return text.strip().title()

def build_graph(relations):
    G = nx.DiGraph()

    for r in relations:
        # ✅ Clean ชื่อ Node ก่อนสร้างกราฟ
        head = clean_node(r["head"])
        tail = clean_node(r["tail"])
        rel_text = r["relation_text"].lower() # Relation ควรเป็นตัวเล็กเสมอ

        G.add_edge(
            head,
            tail,
            relation_text=rel_text,
            sentence=r.get("source_sentence", "")
        )

    return G

def visualize_graph(G, output_file="graph.html"):
    net = Network(height="750px", width="100%", directed=True, bgcolor="#222222", font_color="white")

    # คำนวณ Degree (ใครมีเส้นเชื่อมเยอะสุด ให้ Node ใหญ่สุด)
    degrees = dict(G.degree)

    for node in G.nodes():
        size = 10 + (degrees.get(node, 0) * 5) # ยิ่งเชื่อมเยอะ ยิ่งใหญ่
        net.add_node(node, label=node, title=node, size=size, color="#00ffcc")

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("relation_text", ""), color="#aaaaaa")

    net.show_buttons(filter_=['physics']) # เพิ่มปุ่มปรับแรงดึงดูด
    net.write_html(output_file)
    print(f"📊 Graph saved to {output_file}")