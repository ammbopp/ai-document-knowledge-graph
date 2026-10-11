"""Interactive graph rendering (PyVis + injected UI). Colours follow the web UI theme."""
import html
import json
import textwrap

from pyvis.network import Network

from .graph_layout import tidy_layout

LABEL_WIDTH = 16  # characters per line in node labels

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
    "dark_mode": "Dark mode", "light_mode": "Light mode", "tidy": "Tidy layout",
    "overview": "Overview", "connected": "Connected entities", "no_connections": "No direct connections",
    "no_context": "No additional context.", "type": "Type", "connections": "Connections",
    "implicit": "(Implicit connection to main entity)",
    "zoom_in": "Zoom in", "zoom_out": "Zoom out", "fit": "Fit to screen",
    "importance": "Importance", "showing": "Showing {shown} of {total} relations",
}


def rank_edges(G):
    """
    {(u, v, key): rank} over the real edges, 1 = most important. An edge matters more when its
    endpoints are well connected (hubs, bridges) and when the relation was found repeatedly;
    extraction confidence only breaks ties. Virtual connectors are never ranked.
    """
    real = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if d.get("type") != "virtual"]
    neighbours = {}
    for u, v, _, _ in real:
        neighbours.setdefault(u, set()).add(v)
        neighbours.setdefault(v, set()).add(u)

    def score(edge):
        u, v, _, d = edge
        return len(neighbours[u]) + len(neighbours[v]) + 2 * (d.get("count", 1) - 1) + min(d.get("confidence", 1.0), 1.0)

    ordered = sorted(real, key=lambda e: (-score(e), e[0], e[1], e[2]))
    return {(u, v, k): i for i, (u, v, k, _) in enumerate(ordered, 1)}


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
  .zoom-controls { position: fixed; top: 16px; left: 16px; z-index: 1000; display: flex; flex-direction: column; gap: 8px; }
  .zoom-btn { width: 38px; height: 38px; padding: 0; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: var(--btn-bg); border: 1px solid var(--border-color); color: var(--text-main); cursor: pointer; transition: .15s;
    font-family: inherit; font-size: 20px; line-height: 1; }
  .zoom-btn:hover { background: var(--ink); color: var(--ink-text); border-color: var(--ink); }
  .zoom-btn svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .importance { display: flex; align-items: center; gap: 10px; background: var(--btn-bg); border: 1px solid var(--border-color);
    padding: 6px 16px; border-radius: 999px; font-size: 13px; color: var(--text-main); }
  .importance[hidden] { display: none; }
  .importance input { width: 130px; accent-color: var(--accent); cursor: pointer; }
  .importance .count { color: var(--text-muted); font-size: 12px; white-space: nowrap; min-width: 9.5em; }
</style>
"""

SCRIPT = """
<script type="text/javascript">
  const THEMES = __THEMES__;
  const L = __LABELS__;
  const TYPE_NAMES = __TYPE_NAMES__;
  let isDark = __START_DARK__;
  const EDGE_TOTAL = __EDGE_TOTAL__;   // real relations; the slider keeps the N most important ones
  let tidyPositions = [];

  document.body.insertAdjacentHTML('beforeend', `
    <div class="zoom-controls">
      <button class="zoom-btn" id="zoom-in" title="${L.zoom_in}" aria-label="${L.zoom_in}">
        <svg viewBox="0 0 16 16"><path d="M8 2.5v11M2.5 8h11"/></svg></button>
      <button class="zoom-btn" id="zoom-out" title="${L.zoom_out}" aria-label="${L.zoom_out}">
        <svg viewBox="0 0 16 16"><path d="M2.5 8h11"/></svg></button>
      <button class="zoom-btn" id="zoom-fit" title="${L.fit}" aria-label="${L.fit}">
        <svg viewBox="0 0 16 16"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg></button>
    </div>
    <div class="action-buttons">
      <button id="theme-btn" class="float-btn"></button>
      <button id="tidy-btn" class="float-btn"></button>
      <label class="importance" id="importance" hidden>
        <span>${L.importance}</span>
        <input type="range" id="importance-range" min="1" max="${EDGE_TOTAL}" value="${EDGE_TOTAL}">
        <span class="count" id="importance-count"></span>
      </label>
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
  function zoomBy(factor) {   // zoom around the current view centre
    const scale = Math.min(4, Math.max(0.08, network.getScale() * factor));
    network.moveTo({ scale: scale, animation: { duration: 220, easingFunction: 'easeInOutQuad' } });
  }
  document.getElementById('zoom-in').addEventListener('click', () => zoomBy(1.3));
  document.getElementById('zoom-out').addEventListener('click', () => zoomBy(1 / 1.3));
  // Fit the visible nodes into the canvas, leaving room for the button bar at the bottom
  function fitView(duration) {
    const ids = nodes.get().filter(n => !n.hidden).map(n => n.id);
    if (!ids.length) return;
    const pos = network.getPositions(ids);
    const xs = ids.map(id => pos[id].x), ys = ids.map(id => pos[id].y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const margin = 70, padTop = 24, padBottom = 96, padSide = 24;   // margin: node radius + label, in graph units
    const box = document.getElementById('mynetwork');
    const scale = Math.min(1.4, (box.clientWidth - 2 * padSide) / Math.max(x1 - x0 + 2 * margin, 1),
                                 (box.clientHeight - padTop - padBottom) / Math.max(y1 - y0 + 2 * margin, 1));
    network.moveTo({ position: { x: (x0 + x1) / 2, y: (y0 + y1) / 2 + (padBottom - padTop) / 2 / scale }, scale: scale,
                     animation: duration ? { duration: duration, easingFunction: 'easeInOutQuad' } : false });
  }
  document.getElementById('zoom-fit').addEventListener('click', () => fitView(400));

  document.getElementById('tidy-btn').textContent = L.tidy;
  document.getElementById('tidy-btn').addEventListener('click', () => {   // put every node back where the layout placed it
    tidyPositions.forEach(p => network.moveNode(p.id, p.x, p.y));
    fitView(600);
  });

  // Show only the `limit` most important relations; nodes with no visible relation disappear,
  // the hub stays, and implicit connectors only show while their cluster is visible. Positions never change.
  function applyImportance(limit) {
    const keep = new Set(nodes.get().filter(n => n.entity_type === 'Topic').map(n => n.id));
    edges.get().forEach(e => { if (!e.kg_virtual && e.kg_rank <= limit) { keep.add(e.from); keep.add(e.to); } });
    nodes.update(nodes.get().map(n => ({ id: n.id, hidden: !keep.has(n.id) })));
    edges.update(edges.get().map(e => ({
      id: e.id, hidden: e.kg_virtual ? !(keep.has(e.from) && keep.has(e.to)) : e.kg_rank > limit })));
    document.getElementById('importance-count').textContent =
      L.showing.replace('{shown}', Math.min(limit, EDGE_TOTAL)).replace('{total}', EDGE_TOTAL);
    const selected = network.getSelectedNodes();
    if (selected.length && !keep.has(selected[0])) closePanel();
  }
  const importanceBox = document.getElementById('importance');
  const importanceRange = document.getElementById('importance-range');
  importanceBox.hidden = EDGE_TOTAL < 4;   // nothing worth filtering in a tiny graph
  importanceRange.addEventListener('input', () => applyImportance(+importanceRange.value));
  importanceRange.addEventListener('change', () => fitView(500));   // on release: zoom to what is left

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

    document.getElementById('panel-title').textContent = String(nodeId);   // the id is the unwrapped name
    const type = node.entity_type || 'Entity';
    const theme = isDark ? THEMES.dark : THEMES.light;
    const swatch = (theme.nodes[type] || theme.nodes['Entity']).background;
    const badge = document.getElementById('panel-badge');
    badge.innerHTML = '<span class="info-tag"><i style="background:' + swatch + '"></i></span>';
    badge.firstChild.appendChild(document.createTextNode(TYPE_NAMES[type] || type));
    document.getElementById('panel-overview').innerHTML = node.panel_html || L.no_context;

    const list = document.getElementById('panel-connections');
    list.innerHTML = '';
    const connected = network.getConnectedNodes(nodeId).filter(id => !(nodes.get(id) || {}).hidden);
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
      li.textContent = String(id);   // textContent: article text is never parsed as HTML
      li.addEventListener('click', () => focusAndOpenPanel(id));
      list.appendChild(li);
    });
    document.getElementById('side-panel').classList.add('open');
  }

  setTimeout(function () {
    tidyPositions = nodes.get().map(n => ({ id: n.id, x: n.x, y: n.y }));
    fitView(0);
    applyImportance(EDGE_TOTAL);
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
    positions = tidy_layout(G)
    ranks = rank_edges(G)

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Entity")
        colors = start["nodes"].get(node_type, start["nodes"]["Entity"])
        size = (34 if node_type == "Topic" else 17) + degrees.get(node, 0) * 2.2
        panel = (f"{html.escape(labels['type'])}: {html.escape(type_names.get(node_type, node_type))}<br>"
                 f"{html.escape(labels['connections'])}: {degrees.get(node, 0)}")
        x, y = positions[node]
        net.add_node(
            node, label=textwrap.fill(node, LABEL_WIDTH, break_long_words=False) or node, x=round(x, 1), y=round(y, 1),
            size=size, shape="dot", borderWidth=2, borderWidthSelected=5,
            color={"background": colors["background"], "border": colors["border"],
                   "highlight": {"background": colors["background"], "border": start["highlight"]}},
            font={"color": start["text"], "size": 15, "face": font_face, "vadjust": -2},
            entity_type=node_type, panel_html=panel,
        )

    # Parallel / opposite edges between one pair fan out on alternate sides instead of overlapping
    groups = {}
    for u, v, key in G.edges(keys=True):
        groups.setdefault(frozenset((u, v)), []).append((u, v, key))

    for u, v, key, data in G.edges(keys=True, data=True):
        virtual = data.get("type") == "virtual"
        sentences = [labels["implicit"]] if virtual else data.get("sentences", [])
        siblings = groups[frozenset((u, v))]
        slot = (siblings.index((u, v, key)) - (len(siblings) - 1) / 2) * (1 if u <= v else -1)
        if len(siblings) == 1 or u == v:
            smooth = False
        else:
            smooth = {"enabled": True, "type": "curvedCW" if slot >= 0 else "curvedCCW",
                      "roundness": round(min(0.15 + 0.2 * abs(slot), 0.6), 2)}
        net.add_edge(
            u, v, label="" if virtual else data.get("relation", "").replace("_", " "), smooth=smooth,
            kg_rank=ranks.get((u, v, key), 0), kg_virtual=virtual,
            width=1.6 if virtual else 1.4 + data.get("count", 1) * 0.5,
            title="<br>".join(html.escape(s) for s in sentences) or "",
            color={"color": start["edge_virtual"] if virtual else start["edge"], "highlight": start["edge_highlight"]},
            dashes=virtual, arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            font={"size": 12, "color": start["edge_text"], "align": "horizontal", "strokeWidth": 5,
                  "strokeColor": start["bg"], "face": font_face},
        )

    net.set_options(json.dumps({
        "nodes": {"shadow": {"enabled": True, "color": "rgba(45,55,130,0.18)", "size": 12, "y": 4}},
        "edges": {"smooth": False},
        "physics": {"enabled": False},   # positions come from graph_layout.tidy_layout; drag nodes freely
        "interaction": {"hover": True, "navigationButtons": False, "keyboard": True, "dragNodes": True},
    }))

    page = net.generate_html(output_file) or "<html><body>Error: could not generate graph</body></html>"
    css = CSS.replace("__ACCENT_DARK__", accent[1]).replace("__ACCENT__", accent[0])
    script = (SCRIPT.replace("__THEMES__", json.dumps(themes))
              .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
              .replace("__TYPE_NAMES__", json.dumps(type_names, ensure_ascii=False))
              .replace("__START_DARK__", "true" if dark else "false")
              .replace("__EDGE_TOTAL__", str(len(ranks))))
    page = page.replace("</head>", css + "\n</head>").replace("</body>", script + "\n</body>")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page)
    return page
