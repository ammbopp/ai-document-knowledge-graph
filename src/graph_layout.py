"""Deterministic, tidy node placement for the knowledge graph (no physics jitter, no overlaps)."""
import math

import networkx as nx

SPACING = 110        # px between neighbouring nodes inside a cluster
MIN_DISTANCE = 100   # no two nodes closer than this after relaxation
MIN_EDGE = 170       # shortest edge inside a cluster (room for the relation label)
CLUSTER_GAP = 80    # px between clusters
KAMADA_LIMIT = 150   # Kamada-Kawai is O(n^3); larger clusters use a spring layout


def _real_graph(G):
    """Undirected simple graph of the real relations (virtual connectors are positioned, not used to pull)."""
    R = nx.Graph()
    R.add_nodes_from(sorted(G.nodes))
    R.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("type") != "virtual" and u != v)
    return R


def _layout_cluster(sub):
    """{node: (x, y)} centred on the origin for one connected cluster."""
    n = sub.number_of_nodes()
    if n == 1:
        return {next(iter(sub)): (0.0, 0.0)}
    if n == 2:
        a, b = sorted(sub)
        return {a: (-SPACING * 0.75, 0.0), b: (SPACING * 0.75, 0.0)}
    radius = SPACING * max(1.0, math.sqrt(n)) * 0.75
    try:
        if n > KAMADA_LIMIT:
            raise ValueError
        pos = nx.kamada_kawai_layout(sub, scale=radius)
    except Exception:   # scipy missing, or a big cluster
        pos = nx.spring_layout(sub, seed=7, k=SPACING / max(radius, 1) * 2, iterations=120, scale=radius)
    return _stretch(sub, {node: (float(x), float(y)) for node, (x, y) in pos.items()})


def _stretch(sub, pos):
    """Scale a cluster up until its shortest edge is MIN_EDGE long, so edge labels have room."""
    shortest = min((math.dist(pos[u], pos[v]) for u, v in sub.edges), default=MIN_EDGE)
    factor = MIN_EDGE / shortest if 1e-6 < shortest < MIN_EDGE else 1.0
    return {n: (x * factor, y * factor) for n, (x, y) in pos.items()}


def _relax(pos, min_distance=MIN_DISTANCE, rounds=80):
    """Push apart any pair closer than min_distance (stable order, so the result is repeatable)."""
    nodes = sorted(pos)
    for _ in range(rounds):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                dist = math.hypot(dx, dy)
                if dist >= min_distance:
                    continue
                if dist < 1e-6:                      # identical spots: separate along a fixed direction
                    dx, dy, dist = 1.0, 0.3, math.hypot(1.0, 0.3)
                push = (min_distance - dist) / 2 + 0.5
                ux, uy = dx / dist, dy / dist
                pos[a] = (pos[a][0] - ux * push, pos[a][1] - uy * push)
                pos[b] = (pos[b][0] + ux * push, pos[b][1] + uy * push)
                moved = True
        if not moved:
            break
    return pos


def _centre(pos):
    cx = sum(x for x, _ in pos.values()) / len(pos)
    cy = sum(y for _, y in pos.values()) / len(pos)
    centred = {n: (x - cx, y - cy) for n, (x, y) in pos.items()}
    return centred, max(math.hypot(x, y) for x, y in centred.values())


def tidy_layout(G, hub=None):
    """
    {node: (x, y)} for every node of G. Each connected cluster is laid out on its own
    (Kamada-Kawai: even edge lengths, few crossings); the cluster holding `hub` (default: the
    highest-degree node) sits in the middle and the others are packed on rings around it.
    """
    if G.number_of_nodes() == 0:
        return {}
    real = _real_graph(G)
    if hub is None:
        hub = min(G.nodes, key=lambda n: (-G.degree(n), n))

    clusters = []
    for nodes in nx.connected_components(real):
        pos, radius = _centre(_relax(_layout_cluster(real.subgraph(nodes))))
        clusters.append((hub in nodes, radius, min(nodes), pos))
    clusters.sort(key=lambda c: (not c[0], -c[1], c[2]))   # hub cluster first, then the biggest

    placed = dict(clusters[0][3])
    main_radius = clusters[0][1]
    rest = clusters[1:]
    ring_inner = main_radius + CLUSTER_GAP
    while rest:
        first_radius = rest[0][1] + MIN_DISTANCE / 2
        ring_r = ring_inner + first_radius
        capacity, used, ring = 2 * math.pi * ring_r, 0.0, []
        while rest and (not ring or used + 2 * rest[0][1] + CLUSTER_GAP <= capacity):
            ring.append(rest.pop(0))
            used += 2 * ring[-1][1] + CLUSTER_GAP
        ring_r = max(ring_r, used / (2 * math.pi))
        angle = -math.pi / 2
        for _, radius, _, pos in ring:
            share = (2 * radius + CLUSTER_GAP) / used * 2 * math.pi
            theta = angle + share / 2
            cx, cy = ring_r * math.cos(theta), ring_r * math.sin(theta)
            placed.update({n: (x + cx, y + cy) for n, (x, y) in pos.items()})
            angle += share
        ring_inner = ring_r + max(c[1] for c in ring) + CLUSTER_GAP
    return placed
