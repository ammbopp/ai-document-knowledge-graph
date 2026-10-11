import math

import networkx as nx

from src.graph_builder import build_graph
from src.graph_layout import MIN_DISTANCE, tidy_layout


def rel(h, r, t):
    return {"head": h, "relation": r, "tail": t, "source_sentence": f"{h} {r} {t}."}


def min_gap(pos):
    pts = list(pos.values())
    return min(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])


def test_every_node_is_placed_without_overlap_and_repeatably():
    relations = [rel("Hub", "owns", f"Child {i}") for i in range(12)]
    relations += [rel("Alpha", "met", "Beta"), rel("Beta", "met", "Gamma"), rel("Solo A", "likes", "Solo B")]
    relations += [rel("Hub", "knows", "Child 1"), rel("Child 1", "knows", "Hub")]   # parallel + reverse edge
    G = build_graph(relations)

    pos = tidy_layout(G)
    assert set(pos) == set(G.nodes)
    assert min_gap(pos) >= MIN_DISTANCE - 1
    assert pos == tidy_layout(G)   # deterministic: same graph, same picture


def test_disconnected_clusters_do_not_interleave():
    G = build_graph([rel("A1", "x", "A2"), rel("A2", "x", "A3"), rel("A3", "x", "A1"),
                     rel("B1", "y", "B2"), rel("B2", "y", "B3"), rel("B3", "y", "B1")])
    pos = tidy_layout(G)
    a = [pos[n] for n in ("A1", "A2", "A3")]
    b = [pos[n] for n in ("B1", "B2", "B3")]
    within = max(math.dist(p, q) for grp in (a, b) for p in grp for q in grp)
    between = min(math.dist(p, q) for p in a for q in b)
    assert between > 0.5 * within   # clusters sit apart instead of being mixed together


def test_degenerate_graphs():
    assert tidy_layout(nx.MultiDiGraph()) == {}
    G = nx.MultiDiGraph()
    G.add_node("only")
    assert tidy_layout(G) == {"only": (0.0, 0.0)}
    assert len(tidy_layout(build_graph([rel("A", "to", "B")]))) == 2


def test_edge_importance_favours_hubs_and_repeated_relations():
    from src.graph_view import rank_edges

    relations = [rel("Hub", "owns", f"Leaf {i}") for i in range(5)]
    relations += [rel("Leaf 0", "likes", "Far A"), rel("Far A", "knows", "Far B")]
    relations += [rel("Far A", "knows", "Far B")]   # same relation seen twice -> count 2
    G = build_graph(relations)
    ranks = rank_edges(G)

    assert sorted(ranks.values()) == list(range(1, len(ranks) + 1))   # a total order over real edges only
    assert all(G.edges[e].get("type") != "virtual" for e in ranks)
    hub_ranks = [r for (u, v, _), r in ranks.items() if "Hub" in (u, v)]
    outer = ranks[("Far A", "Far B", "knows")]
    assert max(hub_ranks) < ranks[("Leaf 0", "Far A", "likes")] and outer < ranks[("Leaf 0", "Far A", "likes")]
    assert ranks == rank_edges(G)   # deterministic


def test_graph_page_carries_the_importance_slider(tmp_path):
    from src.graph_view import visualize_graph

    G = build_graph([rel("Hub", "owns", f"Leaf {i}") for i in range(6)])
    page = visualize_graph(G, str(tmp_path / "g.html"), labels={"importance": "ความสำคัญ"})
    assert "importance-range" in page and "ความสำคัญ" in page and "const EDGE_TOTAL = 6;" in page
    assert page.count("kg_rank") >= 6


def test_large_graph_uses_the_fast_layout_and_still_separates_nodes():
    G = nx.MultiDiGraph()
    for i in range(1, 200):
        G.add_edge(f"n{i // 2}", f"n{i}", relation="r")
    pos = tidy_layout(G)
    assert len(pos) == G.number_of_nodes() and min_gap(pos) >= MIN_DISTANCE - 1
