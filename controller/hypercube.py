"""
controller/hypercube.py

Stage 4: dimension-recursive weighted-matching construction of the logical
hypercube topology from pairwise diversity (sqrt-JS) distances, per
Section [Spread-Preserving Hypercube Embedding].

Algorithm (d = log2(n) rounds, n=8 -> d=3):
  Round k merges pairs of "virtual nodes" (sub-cubes of size 2^(k-1)) into
  virtual nodes of size 2^k. The round targets the q_k = (k-1)/(d-1)
  quantile of the round's aggregate virtual-node disagreement D(vn,vn').
  Round 1 (q=0) -> target is the minimum -> minimizes disagreement.
  Round d (q=1) -> target is the maximum -> maximizes disagreement.
  Interior rounds interpolate (for d=3: round 2 targets the median).

  Both (a) WHICH virtual nodes merge (a perfect matching search over the
  current virtual nodes) and (b) the INTERNAL CORRESPONDENCE between
  individual members of a matched pair (searched over the automorphism
  group of the level-(k-1) sub-cube, order (k-1)!*2^(k-1)) are chosen by
  the same rule: minimize total deviation from the round's target. This
  single rule reproduces "minimize/median/maximize" as an emergent special
  case - at q=0 minimizing deviation from the minimum is mathematically
  identical to minimizing total weight (true min-weight perfect matching);
  at q=1, identical to maximizing total weight. Verified directly against
  brute-force minimum-weight matching.

  Only the diversity scores d_ij feed this construction - confidence,
  communication cost, and reliability are Stage-5 selection signals and
  never influence which neighbors a client can reach.

Runs at the CONTROLLER only. Brute-force search (perfect matchings +
automorphisms) is exact and cheap at d=3 (n=8): at most 105 matchings to
check per round, at most 8 automorphisms per merge. This does not scale
to large n - a random-pick strategy would be needed there instead.
"""

import itertools
import numpy as np


def generate_automorphisms(dim: int) -> list:
    """
    All automorphisms of a dim-dimensional hypercube: every combination of
    a bit-position permutation and an XOR flip mask. Order = dim! * 2^dim.
    Returns a list of sigma arrays where sigma[v] = image of local vertex v.
    """
    if dim == 0:
        return [np.array([0])]
    n_vertices = 2 ** dim
    automorphisms = []
    for perm in itertools.permutations(range(dim)):
        for mask in range(n_vertices):
            sigma = np.zeros(n_vertices, dtype=int)
            for v in range(n_vertices):
                bits = [(v >> b) & 1 for b in range(dim)]
                permuted = [bits[perm[b]] for b in range(dim)]
                new_v = sum(permuted[b] << b for b in range(dim))
                new_v ^= mask
                sigma[v] = new_v
            automorphisms.append(sigma)
    return automorphisms


def generate_perfect_matchings(items: list):
    """Yield every perfect matching of an even-length list, as lists of pairs."""
    if len(items) == 0:
        yield []
        return
    first = items[0]
    rest = items[1:]
    for i, other in enumerate(rest):
        pair = (first, other)
        remaining = rest[:i] + rest[i + 1:]
        for sub_matching in generate_perfect_matchings(remaining):
            yield [pair] + sub_matching


def virtual_node_distance(vn_a: tuple, vn_b: tuple, d_matrix: np.ndarray) -> float:
    """D(vn,vn') = sum over all cross pairs of pairwise distance."""
    return sum(d_matrix[u, v] for u in vn_a for v in vn_b)


def build_hypercube_recursive(d_matrix: np.ndarray, verbose: bool = False) -> tuple:
    """
    Dimension-recursive construction. d_matrix is the (n, n) symmetric
    matrix of pairwise sqrt-JS diversity scores between clients (Stage 2-3
    output). n must be a power of 2.

    Returns final_order: a tuple of length n where final_order[vertex] =
    client_index assigned to that hypercube vertex.
    """
    n = d_matrix.shape[0]
    d_rounds = int(np.log2(n))
    assert 2 ** d_rounds == n, "n must be a power of 2"

    virtual_nodes = [(i,) for i in range(n)]

    for k in range(1, d_rounds + 1):
        size = 2 ** (k - 1)
        q_k = (k - 1) / (d_rounds - 1) if d_rounds > 1 else 0.0
        m = len(virtual_nodes)

        pairwise_D = {}
        for a in range(m):
            for b in range(a + 1, m):
                pairwise_D[(a, b)] = virtual_node_distance(virtual_nodes[a], virtual_nodes[b], d_matrix)

        target = np.quantile(list(pairwise_D.values()), q_k)

        if verbose:
            print(f"Round {k}/{d_rounds}: q_k={q_k:.2f}, target D={target:.4f}, "
                  f"{m} virtual nodes of size {size}")

        best_matching, best_score = None, np.inf
        for matching in generate_perfect_matchings(list(range(m))):
            score = sum(abs(pairwise_D[tuple(sorted(pair))] - target) for pair in matching)
            if score < best_score:
                best_score = score
                best_matching = matching

        automorphisms = generate_automorphisms(k - 1)
        new_virtual_nodes = []
        for (a, b) in best_matching:
            vn_a, vn_b = virtual_nodes[a], virtual_nodes[b]
            best_sigma, best_auto_score = None, np.inf
            for sigma in automorphisms:
                edge_distances = [d_matrix[vn_a[i], vn_b[sigma[i]]] for i in range(size)]
                score = sum(abs(ed - target) for ed in edge_distances)
                if score < best_auto_score:
                    best_auto_score = score
                    best_sigma = sigma
            merged = tuple(vn_a) + tuple(vn_b[best_sigma[i]] for i in range(size))
            new_virtual_nodes.append(merged)

        virtual_nodes = new_virtual_nodes

    return virtual_nodes[0]


def hypercube_edges(n_bits: int = 3) -> list:
    """Edge list of an n_bits-dimensional hypercube: vertices differ in exactly one bit."""
    edges = []
    n_vertices = 2 ** n_bits
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            if bin(u ^ v).count("1") == 1:
                edges.append((u, v))
    return edges


def get_neighbor_map(final_order: tuple, edges: list, node_names: list = None) -> dict:
    """
    Convert final_order + edge list into a node_id -> [neighbor_node_ids]
    mapping - what the controller pushes out to each node after building
    the topology.
    """
    n_nodes = len(final_order)
    if node_names is None:
        node_names = [f"node{i}" for i in range(n_nodes)]

    vertex_to_client = {v: c for v, c in enumerate(final_order)}
    client_to_vertex = {c: v for v, c in enumerate(final_order)}

    neighbor_map = {name: [] for name in node_names}
    for u, v in edges:
        client_u, client_v = vertex_to_client[u], vertex_to_client[v]
        neighbor_map[node_names[client_u]].append(node_names[client_v])
        neighbor_map[node_names[client_v]].append(node_names[client_u])

    return neighbor_map


if __name__ == "__main__":
    print("Running self-tests for hypercube.py...\n")

    np.random.seed(3)
    n = 8
    positions = np.sort(np.random.uniform(0, 10, n))
    d_matrix = np.abs(positions[:, None] - positions[None, :])

    best_true_matching, best_true_score = None, np.inf
    for matching in generate_perfect_matchings(list(range(n))):
        score = sum(d_matrix[a, b] for a, b in matching)
        if score < best_true_score:
            best_true_score = score
            best_true_matching = matching

    final_order = build_hypercube_recursive(d_matrix)
    edges = hypercube_edges(3)

    target0 = np.quantile([d_matrix[a, b] for a in range(n) for b in range(a + 1, n)], 0.0)
    best_dev_matching, best_dev_score = None, np.inf
    for matching in generate_perfect_matchings(list(range(n))):
        score = sum(abs(d_matrix[a, b] - target0) for a, b in matching)
        if score < best_dev_score:
            best_dev_score = score
            best_dev_matching = matching
    weight_of_dev_matching = sum(d_matrix[a, b] for a, b in best_dev_matching)
    print(f"Test 1: round-1-style matching weight = {weight_of_dev_matching:.4f}, "
          f"true min-weight = {best_true_score:.4f}")
    assert abs(weight_of_dev_matching - best_true_score) < 1e-9, \
        "FAILED: round-1 logic should be mathematically identical to true min-weight matching"
    print("  PASSED: q=0 deviation-minimization is exactly equivalent to min-weight matching.\n")

    import math
    for dim in [0, 1, 2]:
        autos = generate_automorphisms(dim)
        expected = math.factorial(dim) * (2 ** dim)
        assert len(autos) == expected
    print("Test 2 PASSED: automorphism group sizes match dim!*2^dim exactly (1, 2, 8).\n")

    for trial in range(20):
        np.random.seed(trial)
        positions = np.random.uniform(0, 10, n)
        d_matrix_t = np.abs(positions[:, None] - positions[None, :])
        order = build_hypercube_recursive(d_matrix_t)
        assert sorted(order) == list(range(n)), f"Trial {trial}: invalid permutation"
        c2v = {c: v for v, c in enumerate(order)}
        for client in range(n):
            v = c2v[client]
            neighbors = set()
            for u, w in edges:
                if u == v: neighbors.add(w)
                if w == v: neighbors.add(u)
            assert len(neighbors) == 3, f"Trial {trial}, client {client}: expected 3 neighbors"
    print("Test 3 PASSED: 20/20 random trials produced valid Q3 structures.\n")

    neighbor_map = get_neighbor_map(final_order, edges)
    assert all(len(v) == 3 for v in neighbor_map.values())
    assert len(neighbor_map) == 8
    print("Test 4 PASSED: neighbor map correctly shaped (8 nodes, 3 neighbors each).")

    print("\nAll self-tests passed.")
