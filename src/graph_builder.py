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

def build_graph(relations, root_name=None):
    """
    Build ontology-aware knowledge graph and dynamically find the Main Entity
    """
    G = nx.DiGraph()

    # 1. สร้างกราฟจากความสัมพันธ์ทั้งหมดก่อน (ยังไม่กำหนดศูนย์กลาง)
    for r in relations:
        head = normalize_node(r["head"])
        tail = normalize_node(r["tail"])
        relation = normalize_relation(
            r.get("relation", r.get("relation_text", "related_to"))
        )

        # ---- เพิ่ม Node ----
        if not G.has_node(head):
            G.add_node(head, type=r.get("head_type", "Entity"), confidence=r.get("confidence", 1.0))
        
        if not G.has_node(tail):
            G.add_node(tail, type=r.get("tail_type", "Entity"), confidence=r.get("confidence", 1.0))

        # ---- เพิ่ม/รวม Edge ----
        if G.has_edge(head, tail):
            G[head][tail]["count"] += 1
            if "sentences" not in G[head][tail]:
                G[head][tail]["sentences"] = []
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

    if G.number_of_nodes() == 0:
        return G

    # 2. ค้นหา "ตัวเอก" (Main Entity) โดยดูจาก Node ที่มีเส้นเชื่อมเยอะที่สุด (Degree)
    degrees = dict(G.degree())
    main_entity = max(degrees, key=degrees.get)

    print(f"🌟 AI Identified Main Entity: {main_entity}")

    # อัปเกรดให้ Main Entity กลายเป็นจุดศูนย์กลาง (Topic/Hub)
    G.nodes[main_entity]["type"] = "Topic"
    G.nodes[main_entity]["group"] = "Hub"

    # 3. จับกลุ่มที่ลอยเคว้ง (Disconnected Clusters) มาเชื่อมกับ Main Entity
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    
    for component in components:
        # ถ้ากลุ่มนี้มีตัวเอกอยู่แล้ว ให้ข้ามไป
        if main_entity in component:
            continue
        
        subgraph = G.subgraph(list(component))
        # หาตัวแทนของกลุ่มย่อยนั้น (ตัวที่มีเส้นเชื่อมเยอะสุดในกลุ่ม)
        sorted_nodes = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)
        if not sorted_nodes: continue
        representative_node = sorted_nodes[0][0] 

        # โยงตัวแทนกลุ่มย่อย เข้าหา Main Entity
        G.add_edge(
            main_entity,
            representative_node,
            relation="related_to",  # ใช้คำว่า related_to แทน includes จะดูเป็นธรรมชาติกว่า
            count=1,
            sentences=["(Implicit connection to main entity)"],
            confidence=1.0,
            type="virtual" 
        )

    return G

def visualize_graph(G, output_file="knowledge_graph.html"):
    # ==========================================
    # 1. Base Setup (เริ่มต้นด้วย Light Theme)
    # ==========================================
    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        bgcolor="#f8fafc",     
        font_color="#1e293b",  
        select_menu=False,
        cdn_resources="in_line"
    )

    # สีตั้งต้น (Light Theme)
    COLOR_MAP = {
        "Person": {"background": "#3b82f6", "border": "#2563eb"},       
        "Organization": {"background": "#10b981", "border": "#059669"}, 
        "Location": {"background": "#f59e0b", "border": "#d97706"},     
        "Work": {"background": "#8b5cf6", "border": "#7c3aed"},         
        "Entity": {"background": "#94a3b8", "border": "#64748b"},       
        "Topic": {"background": "#ef4444", "border": "#dc2626"}         
    }

    degrees = dict(G.degree)

    # ==========================================
    # 2. Nodes & Edges Styling
    # ==========================================
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Entity")
        
        base_size = 40 if node_type == "Topic" else 20
        size = base_size + (degrees.get(node, 0) * 2.5) 
        
        colors = COLOR_MAP.get(node_type, COLOR_MAP["Entity"])

        net.add_node(
            node,
            label=node,
            title=f"<b>{node}</b><br>Type: {node_type}<br>Connections: {degrees.get(node, 0)}",
            size=size,
            color={
                "background": colors["background"],
                "border": colors["border"],
                # 🔥 แก้การเลือกโหนด: โหมดสว่างใช้ขอบสีกรมท่าเข้มๆ ให้ชัดเจน
                "highlight": {"background": colors["background"], "border": "#0f172a"}
            },
            # 🔥 แก้สีฟอนต์: โหมดสว่างใช้ตัวอักษรสีเข้ม จะได้ไม่จมพื้นหลัง
            font={"color": "#1e293b", "size": 15, "face": "Segoe UI", "vadjust": -2},
            borderWidth=2,
            borderWidthSelected=6, # 🔥 เพิ่มความหนาขอบตอนเลือกโหนดให้หนาสะใจ
            shape="dot",
            entity_type=node_type
        )

    for u, v, data in G.edges(data=True):
        is_virtual = data.get("type") == "virtual"
        
        net.add_edge(
            u,
            v,
            label=data.get("relation", ""),
            width=2 if is_virtual else (1.5 + (data.get("count", 1) * 0.5)),
            title="<br>".join(data.get("sentences", ["No source context"])),
            color={"color": "#fca5a5" if is_virtual else "#cbd5e1", "highlight": "#94a3b8"}, 
            dashes=True if is_virtual else False,
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            font={
                "size": 12, 
                "color": "#334155", 
                "align": "middle",
                "strokeWidth": 3, 
                "strokeColor": "#f8fafc", # สร้างขอบขาวรอบตัวหนังสือบนเส้น
                "face": "Segoe UI"
            }
        )

    # ==========================================
    # 3. Physics & Smoothness
    # ==========================================
    options = {
        "nodes": {
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.15)", "size": 10}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.015,
                "springLength": 250,
                "springConstant": 0.05,
                "damping": 0.7
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": True, "iterations": 150}
        },
        "interaction": {
            "hover": True,
            "navigationButtons": False, 
            "keyboard": True
        }
    }
    
    net.set_options(json.dumps(options))
    net.write_html(output_file)

    # ==========================================
    # 4. Code Injection (Theme Toggle + JS Logic)
    # ==========================================
    with open(output_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    custom_css = """
    <style>
        :root {
            --panel-bg: rgba(255, 255, 255, 0.95);
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --box-bg: #f8fafc;
            --hover-bg: #e0f2fe;
            --tag-bg: #e0f2fe;
            --tag-text: #0284c7;
            --btn-bg: white;
            --btn-text: #475569;
        }
        
        body.dark-mode {
            --panel-bg: rgba(20, 20, 20, 0.95);
            --text-main: #e0e0e0;
            --text-muted: #9e9e9e;
            --border-color: #333333;
            --box-bg: #2d2d2d;
            --hover-bg: #37474f;
            --tag-bg: #004d40;
            --tag-text: #00e676;
            --btn-bg: #2d2d2d;
            --btn-text: #e0e0e0;
        }

        body, html { margin: 0; padding: 0; overflow: hidden; font-family: 'Segoe UI', sans-serif; transition: background-color 0.4s; }
        
        #side-panel {
            position: fixed; top: 0; right: -400px; width: 360px; height: 100vh;
            background: var(--panel-bg);
            backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
            border-left: 1px solid var(--border-color);
            color: var(--text-main);
            box-shadow: -10px 0 30px rgba(0,0,0,0.1);
            transition: right 0.4s cubic-bezier(0.4, 0, 0.2, 1), background 0.4s, color 0.4s;
            z-index: 9999; padding: 30px 25px; box-sizing: border-box; overflow-y: auto;
        }
        #side-panel.open { right: 0; }
        
        .panel-header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; transition: border-color 0.4s; }
        .panel-header h2 { margin: 0; font-size: 22px; color: var(--text-main); font-weight: 700; transition: color 0.4s; }
        .close-btn { background: var(--box-bg); border: none; color: var(--text-muted); font-size: 20px; cursor: pointer; border-radius: 8px; width: 32px; height: 32px; transition: 0.2s; }
        .close-btn:hover { background: #ef4444; color: white; }
        
        .info-tag { display: inline-block; padding: 6px 14px; border-radius: 20px; background: var(--tag-bg); color: var(--tag-text); font-size: 13px; font-weight: 600; margin-bottom: 20px; transition: 0.4s; }
        
        .panel-section { margin-bottom: 25px; }
        .panel-section h3 { color: var(--text-muted); font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 700; }
        .panel-box { background: var(--box-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 15px; font-size: 14px; line-height: 1.6; color: var(--text-main); transition: 0.4s; }
        
        .connection-list { padding: 0; margin: 0; list-style: none; }
        .connection-item {
            padding: 10px 14px; background: var(--box-bg); border: 1px solid var(--border-color);
            margin-bottom: 8px; border-radius: 8px; font-size: 14px; cursor: pointer;
            transition: all 0.2s; display: flex; align-items: center; color: var(--text-main);
        }
        .connection-item:hover { background: var(--hover-bg); border-color: var(--tag-text); transform: translateX(5px); color: var(--tag-text); }
        .connection-item::before { content: "🔗"; margin-right: 8px; font-size: 12px; }

        .action-buttons {
            position: fixed; bottom: 30px; left: 30px; z-index: 1000;
            display: flex; flex-direction: column; gap: 12px;
        }
        .float-btn {
            background: var(--btn-bg); border: 1px solid var(--border-color); padding: 12px 20px;
            border-radius: 50px; font-family: inherit; font-weight: 600; color: var(--btn-text);
            cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.05); transition: 0.3s;
        }
        .float-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); border-color: #94a3b8; }
    </style>
    """

    custom_script = """
    <script type="text/javascript">
        // Inject Buttons
        const buttonsHTML = `
        <div class="action-buttons">
            <button id="theme-btn" class="float-btn" onclick="toggleTheme()">🌙 Dark Mode</button>
            <button id="freeze-btn" class="float-btn" onclick="togglePhysics()">❄️ Freeze Graph</button>
        </div>
        `;
        document.body.insertAdjacentHTML('beforeend', buttonsHTML);
        
        // Inject Side Panel
        const panelHTML = `
        <div id="side-panel">
            <div class="panel-header">
                <h2 id="panel-title">Node</h2>
                <button class="close-btn" onclick="closePanel()">&times;</button>
            </div>
            <div id="panel-badge"></div>
            <div class="panel-section">
                <h3>Overview</h3>
                <div class="panel-box" id="panel-overview"></div>
            </div>
            <div class="panel-section">
                <h3>Connected Entities</h3>
                <ul class="connection-list" id="panel-connections"></ul>
            </div>
        </div>
        `;
        document.body.insertAdjacentHTML('beforeend', panelHTML);

        // ==========================================
        // Theme Logic
        // ==========================================
        const THEMES = {
            light: {
                bg: "#f8fafc",
                nodes: {
                    "Person": { background: "#3b82f6", border: "#2563eb" },
                    "Organization": { background: "#10b981", border: "#059669" },
                    "Location": { background: "#f59e0b", border: "#d97706" },
                    "Work": { background: "#8b5cf6", border: "#7c3aed" },
                    "Entity": { background: "#94a3b8", border: "#64748b" },
                    "Topic": { background: "#ef4444", border: "#dc2626" }
                },
                edgeReal: "#cbd5e1",
                edgeVirtual: "#fca5a5"
            },
            dark: {
                bg: "#121212",
                nodes: {
                    "Person": { background: "#00E5FF", border: "#00E5FF" },       // Cyan
                    "Organization": { background: "#00E676", border: "#00E676" }, // Neon Green
                    "Location": { background: "#FFEA00", border: "#FFEA00" },     // Yellow
                    "Work": { background: "#FF9100", border: "#FF9100" },         // Orange
                    "Entity": { background: "#B0BEC5", border: "#B0BEC5" },       // Silver
                    "Topic": { background: "#FF3366", border: "#FF3366" }         // Neon Pink
                },
                edgeReal: "#546E7A",
                edgeVirtual: "#FF3366"
            }
        };

        let isDark = false;

        window.toggleTheme = function() {
            isDark = !isDark;
            const theme = isDark ? THEMES.dark : THEMES.light;
            
            // Toggle CSS variables via class
            document.body.classList.toggle('dark-mode', isDark);
            
            // Update Canvas Background
            const container = document.getElementById('mynetwork');
            if(container) container.style.backgroundColor = theme.bg;
            
            // Update Nodes Colors
            const nodeUpdates = nodes.get().map(node => {
                const type = node.entity_type || 'Entity';
                const colors = theme.nodes[type] || theme.nodes['Entity'];
                return {
                    id: node.id,
                    color: {
                        background: colors.background,
                        border: colors.border,
                        // 🔥 แก้ไขการเลือก: ขอบหนาขึ้น และสีขอบจะสว่างทะลุตาในโหมดมืด
                        highlight: { 
                            background: colors.background, 
                            border: isDark ? "#ffffff" : "#0f172a" 
                        }
                    },
                    // 🔥 แก้ไขสีตัวอักษรของ Node: สว่างใช้ตัวดำ มืดใช้ตัวขาว (ตัดกับพื้นหลัง)
                    font: { color: isDark ? "#e2e8f0" : "#1e293b" } 
                };
            });
            nodes.update(nodeUpdates);
            
            // Update Edges Colors
            const edgeUpdates = edges.get().map(edge => {
                const isVirtual = edge.dashes === true;
                return {
                    id: edge.id,
                    color: {
                        color: isVirtual ? theme.edgeVirtual : theme.edgeReal,
                        highlight: isDark ? "#ffffff" : "#94a3b8"
                    },
                    // 🔥 สีตัวอักษรและขอบตัวอักษรบนเส้นเชื่อม
                    font: { 
                        color: isDark ? "#9e9e9e" : "#334155",
                        strokeColor: isDark ? "#121212" : "#f8fafc"
                    } 
                };
            });
            edges.update(edgeUpdates);
            
            // Update Button Text
            document.getElementById('theme-btn').innerHTML = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
        };

        // ==========================================
        // Physics & Panel Logic
        // ==========================================
        let physicsEnabled = true;

        window.togglePhysics = function() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled } });
            document.getElementById('freeze-btn').innerHTML = physicsEnabled ? '❄️ Freeze Graph' : '▶️ Unfreeze Graph';
            if(!physicsEnabled) network.stopSimulation();
        }

        window.closePanel = function() {
            document.getElementById("side-panel").classList.remove("open");
            network.unselectAll(); 
        }

        window.focusAndOpenPanel = function(nodeId) {
            var node = nodes.get(nodeId);
            if(!node) return;

            network.focus(nodeId, {
                scale: 1.2,
                animation: { duration: 800, easingFunction: "easeInOutQuad" }
            });
            network.setSelection({ nodes: [nodeId] }); 

            document.getElementById("panel-title").innerText = node.label || nodeId;
            
            var nodeType = node.entity_type || 'Entity';
            document.getElementById("panel-badge").innerHTML = `<span class="info-tag">${nodeType}</span>`;
            
            if(node.title) {
                document.getElementById("panel-overview").innerHTML = node.title;
            } else {
                document.getElementById("panel-overview").innerHTML = "No additional context.";
            }
            
            var connectedNodes = network.getConnectedNodes(nodeId);
            var connectionsHtml = "";
            
            if(connectedNodes.length > 0) {
                connectedNodes.forEach(function(cNodeId) {
                    var cNode = nodes.get(cNodeId);
                    var cLabel = cNode ? cNode.label : cNodeId;
                    connectionsHtml += `<li class="connection-item" onclick="focusAndOpenPanel(\\'${cNodeId}\\')">${cLabel}</li>`;
                });
            } else {
                connectionsHtml = `<li style="list-style: none; color: var(--text-muted); font-size: 14px;">No direct connections</li>`;
            }
            document.getElementById("panel-connections").innerHTML = connectionsHtml;
            document.getElementById("side-panel").classList.add("open");
        }

        setTimeout(function() {
            if (typeof network !== 'undefined') {
                network.on("click", function (params) {
                    if (params.nodes.length > 0) {
                        focusAndOpenPanel(params.nodes[0]);
                    } else {
                        closePanel(); 
                    }
                });
            }
        }, 1000); 
    </script>
    """

    html_content = html_content.replace('</head>', custom_css + '\n</head>')
    html_content = html_content.replace('</body>', custom_script + '\n</body>')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"📊 Knowledge Graph (Fixed Colors & Selection) saved to {output_file}")