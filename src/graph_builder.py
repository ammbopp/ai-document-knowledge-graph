import networkx as nx
from pyvis.network import Network
import json

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
def build_graph(relations, root_name="Main Topic"):
    """
    Build ontology-aware knowledge graph and force connectivity (Mind Map style)
    """
    G = nx.DiGraph()

    # 1. Add Root Node (Central Topic)
    root_node = normalize_node(root_name)
    G.add_node(
        root_node, 
        type="Topic", 
        confidence=1.0, 
        group="Hub"
    )

    # 2. Add Extracted Relations
    for r in relations:
        head = normalize_node(r["head"])
        tail = normalize_node(r["tail"])
        relation = normalize_relation(
            r.get("relation", r.get("relation_text", "related_to"))
        )

        # ---- Node attributes ----
        if not G.has_node(head):
            G.add_node(head, type=r.get("head_type", "Entity"), confidence=r.get("confidence", 1.0))
        
        if not G.has_node(tail):
            G.add_node(tail, type=r.get("tail_type", "Entity"), confidence=r.get("confidence", 1.0))

        # ---- Edge merge logic ----
        if G.has_edge(head, tail):
            G[head][tail]["count"] += 1
            G[head][tail]["sentences"].append(r.get("source_sentence", ""))
        else:
            G.add_edge(
                head,
                tail,
                relation=relation,
                count=1,
                sentences=[r.get("source_sentence", "")],
                confidence=r.get("confidence", 1.0)
            )

    # 3. Force Connectivity (Mind Map Logic)
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    
    print(f"🔗 Found {len(components)} disconnected clusters. Merging into Mind Map...")

    for component in components:
        if root_node in component:
            continue
        
        subgraph = G.subgraph(list(component))
        # Sort nodes by degree to find the most "central" node in the cluster
        sorted_nodes = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)
        
        if not sorted_nodes:
            continue

        representative_node = sorted_nodes[0][0] 

        G.add_edge(
            root_node,
            representative_node,
            relation="includes", 
            count=1,
            sentences=["(Implicit connection to main topic)"],
            confidence=1.0,
            type="virtual" 
        )

    return G

def visualize_graph(G, output_file="knowledge_graph.html"):
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#0f172a",
        font_color="white",
        select_menu=True,
        cdn_resources="in_line"
        # filter_menu=True  <-- เอาออกถ้าทำให้ error ในบาง env
    )

    # Color by entity type
    COLOR_MAP = {
        "Person": "#60a5fa",       # Blue
        "Organization": "#34d399", # Green
        "Location": "#fbbf24",     # Yellow
        "Work": "#f472b6",         # Pink
        "Entity": "#94a3b8",       # Gray
        "Topic": "#ef4444"         # Red (Root)
    }

    degrees = dict(G.degree)

    # ---- Nodes ----
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Entity")
        
        base_size = 40 if node_type == "Topic" else 15
        size = base_size + degrees.get(node, 0) * 3
        
        color = COLOR_MAP.get(node_type, "#94a3b8")

        net.add_node(
            node,
            label=node,
            title=f"<b>{node}</b><br>Type: {node_type}",
            size=size,
            color=color,
            borderWidth=3 if node_type == "Topic" else 1,
            borderWidthSelected=4
        )

    # ---- Edges ----
    for u, v, data in G.edges(data=True):
        is_virtual = data.get("type") == "virtual"
        
        net.add_edge(
            u,
            v,
            label=data.get("relation"),
            width=1 if is_virtual else (1 + data.get("count", 1)),
            title="<br>".join(data.get("sentences", [])),
            color="#ef4444" if is_virtual else "#cbd5f5", 
            dashes=True if is_virtual else False,
            arrows="to"
        )

    # Physics Options (Mind Map Style)
    # ใช้ json.dumps เพื่อความชัวร์เรื่อง format
    options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -150,
                "centralGravity": 0.02,
                "springLength": 150,
                "springConstant": 0.1,
                "damping": 0.4
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": True,
                "iterations": 200,
                "updateInterval": 50
            }
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True
        }
    }
    
    net.set_options(json.dumps(options))

    # ❌ REMOVED: net.show_buttons(filter_=["physics"]) 
    # สาเหตุ: ขัดแย้งกับ set_options ใน pyvis บางเวอร์ชัน ทำให้เกิด AttributeError

    net.write_html(output_file)
    print(f"📊 Knowledge Graph (Mind Map) saved to {output_file}")