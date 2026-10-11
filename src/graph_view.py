"""Interactive graph rendering (PyVis + injected UI). Colours follow the web UI theme."""
import html
import json

from pyvis.network import Network

# Pastel node colours: (background, border) per entity type. "Topic" (hub) uses the theme accent.
_LIGHT = {
    "Person": ("#8aa6ff", "#5f80ee"),
    "Organization": ("#5ed3b3", "#38b394"),
    "Location": ("#ffbb78", "#f09a45"),
    "Time": ("#ff9cc2", "#ef6f9f"),
    "Work": ("#bda6ff", "#9c80f2"),
    "Entity": ("#bcc5d8", "#98a3bb"),
}
_DARK = {
    "Person": ("#9db4ff", "#9db4ff"),
    "Organization": ("#6fe3c4", "#6fe3c4"),
    "Location": ("#ffc88c", "#ffc88c"),
    "Time": ("#ffa9cb", "#ffa9cb"),
    "Work": ("#cdbbff", "#cdbbff"),
    "Entity": ("#8f99ae", "#8f99ae"),
}
NODE_TYPES = ["Person", "Organization", "Location", "Time", "Work", "Entity", "Topic"]

SURFACE = {
    "light": {"bg": "#fbfbfe", "text": "#262a3d", "edge": "#c6cce0", "edge_virtual": "#f2a3a3",
              "edge_text": "#4d5472", "highlight": "#14161f"},
    "dark": {"bg": "#12141c", "text": "#e6e8f2", "edge": "#4b5269", "edge_virtual": "#ff7a8a",
             "edge_text": "#a3a9bd", "highlight": "#ffffff"},
}

DEFAULT_LABELS = {
    "dark_mode": "Dark mode", "light_mode": "Light mode", "freeze": "Freeze layout", "unfreeze": "Resume layout",
    "overview": "Overview", "connected": "Connected entities", "no_connections": "No direct connections",
    "no_context": "No additional context.", "type": "Type", "connections": "Connections",
    "implicit": "(Implicit connection to main entity)",
}


def node_colors(accent, mode="light"):
    """{type: {"background", "border"}} for one mode; accent is that mode's accent hex."""
    base = _DARK if mode == "dark" else _LIGHT
    colors = {t: {"background": bg, "border": bd} for t, (bg, bd) in base.items()}
    colors["Topic"] = {"background": accent, "border": accent}
    return colors


def _themes(accent_light, accent_dark):
    out = {}
    for mode, accent in (("light", accent_light), ("dark", accent_dark)):
        out[mode] = {**SURFACE[mode], "nodes": node_colors(accent, mode), "edge_highlight": accent}
    return out


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600&display=swap');
  :root { --panel-bg: rgba(255,255,255,.94); --text-main: #262a3d; --text-muted: #7a8099; --border-color: #e4e7f1;
          --box-bg: #f5f6fb; --hover-bg: #eceefa; --btn-bg: #ffffff; --ink: #14161f; --ink-text: #ffffff;
          --accent: __ACCENT__; }
  body.dark-mode { --panel-bg: rgba(24,27,38,.94); --text-main: #e6e8f2; --text-muted: #9aa1b8; --border-color: #2b3042;
                   --box-bg: #1d2130; --hover-bg: #262b3d; --btn-bg: #1d2130; --ink: #eceef6; --ink-text: #12141c;
                   --accent: __ACCENT_DARK__; }
  body, html { margin: 0; padding: 0; overflow: hidden; font-family: 'Prompt','Noto Sans Thai','Segoe UI',sans-serif; }
  #side-panel { position: fixed; top: 12px; right: -420px; width: 340px; height: calc(100vh - 24px);
    background: var(--panel-bg); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border-color); border-radius: 20px; color: var(--text-main);
    box-shadow: 0 24px 60px -28px rgba(30,40,110,.45); transition: right .35s cubic-bezier(.4,0,.2,1);
    z-index: 9999; padding: 24px 22px; box-sizing: border-box; overflow-y: auto; }
  #side-panel.open { right: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 14px; }
  .panel-header h2 { margin: 0; font-size: 20px; font-weight: 500; line-height: 1.35; word-break: break-word; }
  .close-btn { flex: none; background: var(--box-bg); border: 1px solid var(--border-color); color: var(--text-muted);
    font-size: 18px; cursor: pointer; border-radius: 50%; width: 30px; height: 30px; line-height: 1; }
  .close-btn:hover { background: var(--ink); color: var(--ink-text); }
  .info-tag { display: inline-flex; align-items: center; gap: 7px; padding: 4px 12px; border-radius: 999px;
    background: var(--box-bg); border: 1px solid var(--border-color); font-size: 12.5px; margin-bottom: 18px; }
  .info-tag i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .panel-section { margin-bottom: 22px; }
  .panel-section h3 { color: var(--text-muted); font-size: 12px; font-weight: 500; letter-spacing: .04em; margin: 0 0 8px; }
  .panel-box { background: var(--box-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 12px 14px;
    font-size: 13.5px; line-height: 1.6; }
  .connection-list { padding: 0; margin: 0; list-style: none; }
  .connection-item { padding: 9px 14px; background: var(--box-bg); border: 1px solid var(--border-color); margin-bottom: 7px;
    border-radius: 12px; font-size: 13.5px; cursor: pointer; transition: .15s; }
  .connection-item:hover { background: var(--hover-bg); border-color: var(--accent); color: var(--accent); }
  .action-buttons { position: fixed; bottom: 16px; left: 16px; z-index: 1000; display: flex; gap: 8px; }
  .float-btn { background: var(--btn-bg); border: 1px solid var(--border-color); padding: 8px 16px; border-radius: 999px;
    font-family: inherit; font-size: 13px; font-weight: 500; color: var(--text-main); cursor: pointer; transition: .15s; }
  .float-btn:hover { border-color: var(--ink); }
</style>
"""

SCRIPT = """
<script type="text/javascript">
  const THEMES = __THEMES__;
  const L = __LABELS__;
  const TYPE_NAMES = __TYPE_NAMES__;
  let isDark = __START_DARK__;
  let physicsEnabled = true;

  document.body.insertAdjacentHTML('beforeend', `
    <div class="action-buttons">
      <button id="theme-btn" class="float-btn"></button>
      <button id="freeze-btn" class="float-btn"></button>
    </div>
    <div id="side-panel">
      <div class="panel-header"><h2 id="panel-title"></h2><button class="close-btn" id="close-btn">&times;</button></div>
      <div id="panel-badge"></div>
      <div class="panel-section"><h3>${L.overview}</h3><div class="panel-box" id="panel-overview"></div></div>
      <div class="panel-section"><h3>${L.connected}</h3><ul class="connection-list" id="panel-connections"></ul></div>
    </div>`);

  function applyTheme() {
    const theme = isDark ? THEMES.dark : THEMES.light;
    document.body.classList.toggle('dark-mode', isDark);
    document.body.style.backgroundColor = theme.bg;
    const container = document.getElementById('mynetwork');
    if (container) container.style.backgroundColor = theme.bg;

    nodes.update(nodes.get().map(node => {
      const c = theme.nodes[node.entity_type] || theme.nodes['Entity'];
      return { id: node.id,
               color: { background: c.background, border: c.border,
                        highlight: { background: c.background, border: theme.highlight } },
               font: { color: theme.text } };
    }));
    edges.update(edges.get().map(edge => ({
      id: edge.id,
      color: { color: edge.dashes === true ? theme.edge_virtual : theme.edge, highlight: theme.edge_highlight },
      font: { color: theme.edge_text, strokeColor: theme.bg }
    })));
    document.getElementById('theme-btn').textContent = isDark ? L.light_mode : L.dark_mode;
  }

  document.getElementById('theme-btn').addEventListener('click', () => { isDark = !isDark; applyTheme(); });
  document.getElementById('freeze-btn').textContent = L.freeze;
  document.getElementById('freeze-btn').addEventListener('click', () => {
    physicsEnabled = !physicsEnabled;
    network.setOptions({ physics: { enabled: physicsEnabled } });
    document.getElementById('freeze-btn').textContent = physicsEnabled ? L.freeze : L.unfreeze;
    if (!physicsEnabled) network.stopSimulation();
  });

  function closePanel() {
    document.getElementById('side-panel').classList.remove('open');
    network.unselectAll();
  }
  document.getElementById('close-btn').addEventListener('click', closePanel);

  function focusAndOpenPanel(nodeId) {
    const node = nodes.get(nodeId);
    if (!node) return;
    network.focus(nodeId, { scale: 1.2, animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
    network.setSelection({ nodes: [nodeId] });

    document.getElementById('panel-title').textContent = node.label || nodeId;
    const type = node.entity_type || 'Entity';
    const theme = isDark ? THEMES.dark : THEMES.light;
    const swatch = (theme.nodes[type] || theme.nodes['Entity']).background;
    const badge = document.getElementById('panel-badge');
    badge.innerHTML = '<span class="info-tag"><i style="background:' + swatch + '"></i></span>';
    badge.firstChild.appendChild(document.createTextNode(TYPE_NAMES[type] || type));
    document.getElementById('panel-overview').innerHTML = node.panel_html || L.no_context;

    const list = document.getElementById('panel-connections');
    list.innerHTML = '';
    const connected = network.getConnectedNodes(nodeId);
    if (connected.length === 0) {
      const li = document.createElement('li');
      li.style.cssText = 'list-style:none;color:var(--text-muted);font-size:13.5px';
      li.textContent = L.no_connections;
      list.appendChild(li);
    }
    connected.forEach(id => {
      const other = nodes.get(id);
      const li = document.createElement('li');
      li.className = 'connection-item';
      li.textContent = other ? other.label : id;   // textContent: article text is never parsed as HTML
      li.addEventListener('click', () => focusAndOpenPanel(id));
      list.appendChild(li);
    });
    document.getElementById('side-panel').classList.add('open');
  }

  setTimeout(function () {
    applyTheme();
    network.on('click', params => params.nodes.length > 0 ? focusAndOpenPanel(params.nodes[0]) : closePanel());
  }, 700);
</script>
"""


def visualize_graph(G, output_file="knowledge_graph.html", *, accent=("#6b63d9", "#a9a3ff"),
                    labels=None, type_names=None, dark=False):
    """
    Write an interactive HTML graph.
    accent: (light, dark) accent hex used for the hub node, hover and highlights.
    labels / type_names: UI strings (see DEFAULT_LABELS); dark: initial mode.
    """
    labels = {**DEFAULT_LABELS, **(labels or {})}
    type_names = {t: t for t in NODE_TYPES} | (type_names or {})
    themes = _themes(*accent)
    start = themes["dark" if dark else "light"]

    net = Network(height="100vh", width="100%", directed=True, bgcolor=start["bg"], font_color=start["text"],
                  select_menu=False, cdn_resources="in_line")
    degrees = dict(G.degree)
    font_face = "Prompt, 'Noto Sans Thai', 'Segoe UI', sans-serif"

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Entity")
        colors = start["nodes"].get(node_type, start["nodes"]["Entity"])
        size = (34 if node_type == "Topic" else 17) + degrees.get(node, 0) * 2.2
        panel = (f"{html.escape(labels['type'])}: {html.escape(type_names.get(node_type, node_type))}<br>"
                 f"{html.escape(labels['connections'])}: {degrees.get(node, 0)}")
        net.add_node(
            node, label=node, size=size, shape="dot", borderWidth=2, borderWidthSelected=5,
            color={"background": colors["background"], "border": colors["border"],
                   "highlight": {"background": colors["background"], "border": start["highlight"]}},
            font={"color": start["text"], "size": 14, "face": font_face, "vadjust": -2},
            entity_type=node_type, panel_html=panel,
        )

    for u, v, data in G.edges(data=True):
        virtual = data.get("type") == "virtual"
        sentences = [labels["implicit"]] if virtual else data.get("sentences", [])
        net.add_edge(
            u, v, label=data.get("relation", "").replace("_", " "),
            width=1.6 if virtual else 1.4 + data.get("count", 1) * 0.5,
            title="<br>".join(html.escape(s) for s in sentences) or "",
            color={"color": start["edge_virtual"] if virtual else start["edge"], "highlight": start["edge_highlight"]},
            dashes=virtual, arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            font={"size": 11, "color": start["edge_text"], "align": "middle", "strokeWidth": 4,
                  "strokeColor": start["bg"], "face": font_face},
        )

    net.set_options(json.dumps({
        "nodes": {"shadow": {"enabled": True, "color": "rgba(45,55,130,0.18)", "size": 12, "y": 4}},
        "edges": {"smooth": {"type": "continuous"}},
        "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -120, "centralGravity": 0.015, "springLength": 250,
                                 "springConstant": 0.05, "damping": 0.7},
            "solver": "forceAtlas2Based", "stabilization": {"enabled": True, "iterations": 150},
        },
        "interaction": {"hover": True, "navigationButtons": False, "keyboard": True},
    }))

    page = net.generate_html(output_file) or "<html><body>Error: could not generate graph</body></html>"
    css = CSS.replace("__ACCENT_DARK__", accent[1]).replace("__ACCENT__", accent[0])
    script = (SCRIPT.replace("__THEMES__", json.dumps(themes))
              .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
              .replace("__TYPE_NAMES__", json.dumps(type_names, ensure_ascii=False))
              .replace("__START_DARK__", "true" if dark else "false"))
    page = page.replace("</head>", css + "\n</head>").replace("</body>", script + "\n</body>")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page)
    return page
