"""
Interactive graph rendering (PyVis + injected UI). Colours follow the web UI theme.
The viewer's styles and behaviour live in assets/graph/viewer.css and viewer.js.
"""
import html
import json
import textwrap
from functools import lru_cache
from pathlib import Path

from pyvis.network import Network

from .graph_layout import tidy_layout

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "graph"
LABEL_WIDTH = 16  # characters per line in node labels


@lru_cache(maxsize=None)
def _asset(name):
    return (ASSET_DIR / name).read_text(encoding="utf-8")

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
    css = _asset("viewer.css").replace("__ACCENT_DARK__", accent[1]).replace("__ACCENT__", accent[0])
    script = (_asset("viewer.js").replace("__THEMES__", json.dumps(themes))
              .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
              .replace("__TYPE_NAMES__", json.dumps(type_names, ensure_ascii=False))
              .replace("__START_DARK__", "true" if dark else "false")
              .replace("__EDGE_TOTAL__", str(len(ranks))))
    page = page.replace("</head>", f"\n<style>\n{css}</style>\n\n</head>").replace(
        "</body>", f'\n<script type="text/javascript">\n{script}</script>\n\n</body>')
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page)
    return page
