"""
controller/periodic_refresh.py

Stage 8: periodic refresh.

Every R rounds (not every round), clients re-score peers using accumulated
training signal rather than the Stage-1 snapshot, and the hypercube
assignment is rewired if the refreshed scores warrant it. This keeps the
topology adapted to how client models have diverged or converged since
setup, without paying the full O(n^2*m*K) re-scoring cost every round.

This module runs at the CONTROLLER (matching controller/hypercube.py's
scope note), reusing:
  - node/logit_exchange.py's get_reference_batch / load_this_node_model /
    compute_logits to regenerate each node's CURRENT logits from its
    latest saved weights (not the frozen Stage-1 snapshot);
  - node/peer_scoring.py's compute_diversity_score for sqrt-JS d_ij;
  - controller/hypercube.py's build_hypercube_recursive / hypercube_edges /
    get_neighbor_map -- the EXACT SAME embedding construction Stage 4
    used, so a rewire produces a topology built the same way, just from
    fresher scores.

IMPORTANT SCOPE NOTE: the paper says the hypercube is "re-optimized only
if compatibility drift exceeds a threshold" (Section III-A) but gives no
formula for that threshold. R (refresh period) and the drift threshold
below are STATIC PLACEHOLDERS -- a design choice documented as such, not
values derived from the paper's math, same pattern as the phase-control
schedule in node/phase_control.py.

IMPORTANT SCOPE NOTE: this module only refreshes the DIVERSITY-based
topology (the d_ij matrix and, if warranted, the rewired hypercube). It
does NOT yet implement real reliability tracking -- peer_scoring.py's
reliability_score() remains its Stage-2/3 stub, explicitly waiting on
"round-level history" that this module could in principle start providing
(participation/consistency across rounds). That is deliberately left as a
separate follow-up pass, not bundled into this one, so the core
refresh/rewire mechanism can be verified in isolation first.
"""

import copy
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node.logit_exchange import get_reference_batch, compute_logits, load_this_node_model
from node.peer_scoring import compute_diversity_score
from controller.hypercube import build_hypercube_recursive, hypercube_edges, get_neighbor_map


# ---- Config / topology I/O -----------------------------------------------------

def load_config(path: str = "configs/periodic_refresh_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_topology(path: str = "results/topology.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_topology(topology: dict, path: str = "results/topology.json"):
    with open(path, "w") as f:
        json.dump(topology, f, indent=2)


# ---- Stage 8 core: should we even refresh this round? --------------------------

def should_refresh(round_num: int, R: int) -> bool:
    """Algorithm 1, line 11: if t mod R == 0, re-score peers; rewire if warranted."""
    return round_num % R == 0


# ---- Recompute the full pairwise diversity matrix from CURRENT weights ---------

def compute_full_diversity_matrix(cfg: dict, node_names: list) -> np.ndarray:
    """
    Recompute d_ij for every pair of the n nodes, using each node's most
    recently saved weights (logs/weights/{node_id}.pt) -- i.e. reflecting
    accumulated training/distillation since setup, not the frozen Stage-1
    snapshot used to originally build the hypercube.
    """
    n = len(node_names)
    reference_images = get_reference_batch(cfg["data_root"], cfg["reference_batch_size"])

    logits_by_node = {}
    for node_id in node_names:
        model = load_this_node_model(cfg, node_id)
        logits_by_node[node_id] = compute_logits(model, reference_images)

    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = compute_diversity_score(logits_by_node[node_names[i]], logits_by_node[node_names[j]])
            d_matrix[i, j] = d_ij
            d_matrix[j, i] = d_ij

    return d_matrix


# ---- Drift metric ----------------------------------------------------------------

def compute_drift(old_matrix: np.ndarray, new_matrix: np.ndarray) -> float:
    """
    Mean absolute change in d_ij across all pairs (upper triangle, since
    the matrix is symmetric with a zero diagonal). This is the
    "compatibility drift" the paper references without giving a formula
    (Section III-A) -- a design choice, documented as such.
    """
    n = old_matrix.shape[0]
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(new_matrix[i, j] - old_matrix[i, j]))
    return float(np.mean(diffs))


# ---- Stage 8 core: refresh + conditional rewire ----------------------------------

def refresh_topology(round_num: int, cfg: dict, topology_path: str = "results/topology.json") -> dict:
    """
    Run one Stage 8 refresh event at round `round_num`:
      1. Recompute the full diversity matrix from current node weights.
      2. Compute drift against the matrix currently stored in topology.json.
      3. If drift > cfg['drift_threshold']: rebuild the hypercube (same
         construction as Stage 4) from the fresh matrix, and overwrite
         neighbor_map/final_order/diversity_matrix.
         Else: keep the existing neighbor_map/final_order (topology stays
         fixed), but still update the stored diversity_matrix so scores
         don't go stale even without a rewire.

    Returns a summary dict: {drift, threshold, rewired, old_neighbor_map,
    new_neighbor_map} for inspection/logging. Always writes the updated
    topology.json (diversity_matrix is refreshed either way).
    """
    old_topology = load_topology(topology_path)
    node_names = old_topology["node_names"]
    old_matrix = np.array(old_topology["diversity_matrix"])

    new_matrix = compute_full_diversity_matrix(cfg, node_names)
    drift = compute_drift(old_matrix, new_matrix)
    threshold = cfg["drift_threshold"]
    rewired = drift > threshold

    new_topology = copy.deepcopy(old_topology)
    new_topology["diversity_matrix"] = new_matrix.tolist()
    new_topology["last_refreshed_round"] = round_num
    new_topology["last_drift"] = drift
    new_topology["last_rewired"] = rewired

    old_neighbor_map = old_topology["neighbor_map"]
    new_neighbor_map = old_neighbor_map

    if rewired:
        final_order = build_hypercube_recursive(new_matrix)
        edges = hypercube_edges(int(np.log2(len(node_names))))
        new_neighbor_map = get_neighbor_map(final_order, edges, node_names)
        new_topology["neighbor_map"] = new_neighbor_map
        new_topology["final_order"] = list(final_order)

    save_topology(new_topology, topology_path)

    return {
        "round": round_num,
        "drift": drift,
        "threshold": threshold,
        "rewired": rewired,
        "old_neighbor_map": old_neighbor_map,
        "new_neighbor_map": new_neighbor_map,
    }


# ---- Standalone test / inspection -----------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    test_round = int(os.environ.get("ROUND", str(cfg["R"])))

    print(f"Running Stage 8 periodic refresh check for round {test_round}...")
    print(f"R={cfg['R']}  drift_threshold={cfg['drift_threshold']}\n")

    if not should_refresh(test_round, cfg["R"]):
        print(f"Round {test_round} is not a refresh round (round mod R != 0). "
              f"Nothing to do -- topology.json left untouched.")
        sys.exit(0)

    result = refresh_topology(test_round, cfg)

    print(f"Drift = {result['drift']:.4f}  (threshold = {result['threshold']:.4f})")
    print(f"Rewired = {result['rewired']}\n")

    if result["rewired"]:
        print("Neighbor map CHANGED:")
        for node_id in result["old_neighbor_map"]:
            old_n = sorted(result["old_neighbor_map"][node_id])
            new_n = sorted(result["new_neighbor_map"][node_id])
            flag = "  <-- changed" if old_n != new_n else ""
            print(f"  {node_id}: {old_n} -> {new_n}{flag}")
    else:
        print("Neighbor map UNCHANGED (drift below threshold, topology stays fixed). "
              "diversity_matrix in topology.json was still updated with fresh scores.")
