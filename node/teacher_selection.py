"""
node/teacher_selection.py

Stage 5: teacher selection via the Critical Disagreement Band (CDB) filter.

At round t, client i restricts attention to the subset of its hypercube
neighbors N(i) whose current disagreement falls inside the active band:

    F_i^t = { j in N(i) : d_ij in [d_min^t, d_max^t] }

If F_i^t is non-empty, i picks the highest-utility neighbor as its teacher:

    j* = argmax_{j in F_i^t} U_i(j)

using U_i(j) = w_u*util_i(j) + w_c*conf(j) + w_cost^t*cost_score(i,j) +
w_r*rel(j)   (Eq. 3). Otherwise i skips distillation this round.

STAGE 7 INTEGRATION: d_min^t, d_max^t, and w_cost^t are now obtained from
node/phase_control.py's get_phase_params(round_num, T, phase_cfg), which
implements the paper's phase-aware schedule (Section IV-C). w_u, w_c, w_r
remain fixed constants (only w_cost^t is phase-adaptive per the paper),
read from configs/teacher_selection_config.yaml.

ROUND-SCOPED LOGITS CACHE: select_teacher() accepts an optional
logits_cache dict. When the orchestration loop (controller/run_experiment.py)
builds one fresh empty cache at the START of each round (before any node
has trained that round) and passes the SAME dict to every node's call this
round, every node scores against a consistent round-start snapshot of all
8 nodes' weights -- matching Algorithm 1's "for each client i (parallel)
do" semantics -- rather than a sequential loop where an earlier node's
in-round update could leak into a later node's neighbor lookup. This also
avoids redundant disk loads/forward passes when multiple nodes share a
neighbor. When logits_cache=None (e.g. standalone single-node testing via
`python node/teacher_selection.py`), behavior is unchanged from before:
every call loads fresh from disk, no caching.

IMPORTANT NOTE ON d_ij: rather than reading the frozen setup-time
diversity_matrix from results/topology.json, this module recomputes d_ij
LIVE for the current round via peer_scoring.compute_cdb_factors, using
freshly generated logits. This matches the paper's intent: Stage 5 filters
on the round's CURRENT disagreement, whereas the frozen snapshot in
topology.json was only ever meant to seed the fixed hypercube embedding
(Stage 4), not to drive round-by-round teacher selection. Only N(i) itself
(the fixed neighbor set) is read from topology.json.

IMPORTANT NOTE ON COST: peer_scoring.py's communication_cost() returns a
raw hop-based cost where HIGHER = more expensive. The paper's Eq. (3)
expects cost(i,j) oriented so HIGHER = cheaper, normalized to [0,1], with
a non-negative weight w_cost^t (Table I, A5). This module performs that
normalization here -- peer_scoring.py itself is left untouched.
"""

import json
import os
import sys

import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node.logit_exchange import get_reference_batch, compute_logits, load_this_node_model
from node.peer_scoring import compute_cdb_factors
from node import phase_control


# ---- Config / topology loading ---------------------------------------------

def load_config(path: str = "configs/teacher_selection_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_topology(path: str = "results/topology.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---- Round-scoped logits cache ------------------------------------------------

def get_node_logits(node_id: str, cfg: dict, reference_images: torch.Tensor,
                     logits_cache: dict = None) -> torch.Tensor:
    """
    Return node_id's logits on reference_images. If logits_cache is
    provided and already has an entry for node_id, reuse it (no disk load,
    no forward pass). Otherwise compute fresh (loading node_id's current
    saved weights) and, if a cache dict was provided, store the result so
    later calls this round reuse it.
    """
    if logits_cache is not None and node_id in logits_cache:
        return logits_cache[node_id]

    model = load_this_node_model(cfg, node_id)
    logits = compute_logits(model, reference_images)

    if logits_cache is not None:
        logits_cache[node_id] = logits

    return logits


# ---- Cost normalization ------------------------------------------------------

def cost_to_cheapness_score(raw_cost: float, max_hops: int) -> float:
    """
    Convert peer_scoring.py's raw hop-based cost (higher = more expensive)
    into the paper's Eq. (3) convention: a [0,1]-normalized score where
    higher = cheaper. 1.0 = zero hops (free), 0.0 = at the hypercube
    diameter (max_hops). Clamped to [0,1] in case raw_cost exceeds
    max_hops for any reason.
    """
    score = 1.0 - (raw_cost / max_hops)
    return max(0.0, min(1.0, score))


# ---- Eq. (3): U_i(j) ----------------------------------------------------------

def compute_utility(factors: dict, cost_score: float, cfg: dict, w_cost: float) -> float:
    """
    U_i(j) = w_u*util_i(j) + w_c*conf(j) + w_cost*cost_score + w_r*rel(j)

    `factors` is the dict returned by peer_scoring.compute_cdb_factors
    (keys: diversity, utility, confidence, communication_cost, reliability).
    Note diversity is deliberately NOT part of U_i(j) -- per the paper, it
    only gates eligibility via the CDB band (see select_teacher below).

    w_u, w_c, w_r come from cfg (fixed constants). w_cost is passed in
    explicitly since it is phase-adaptive (Stage 7, w_cost^t) rather than
    a flat config value.
    """
    return (
        cfg["w_u"] * factors["utility"]
        + cfg["w_c"] * factors["confidence"]
        + w_cost * cost_score
        + cfg["w_r"] * factors["reliability"]
    )


# ---- Stage 5 core: band filter + argmax selection -----------------------------

def select_teacher(node_id: str, cfg: dict, topology: dict, round_num: int = 1,
                    hop_distances: dict = None, logits_cache: dict = None,
                    reference_images: torch.Tensor = None, T: int = None):
    """
    Run Stage 5 for a single client `node_id` at round `round_num`.

    d_min^t, d_max^t, and w_cost^t are obtained from Stage 7's phase
    schedule (node/phase_control.py) for this round_num, out of the total
    T configured in configs/phase_control_config.yaml.

    hop_distances: optional dict {peer_node_id: hop_distance}. Defaults to
    cfg["hop_distance_default"] for every neighbor if not provided.

    logits_cache: optional dict for round-scoped logits reuse (see module
    docstring). If None, every node's logits are loaded fresh from disk
    (safe default for standalone/isolated calls).

    reference_images: optional pre-loaded reference batch. If None,
    regenerated via get_reference_batch (deterministic, so safe either
    way, just avoids redundant reloading when called many times per round).

    Returns (j_star, F_i_t, per_neighbor_info, phase_params).
    """
    neighbor_ids = topology["neighbor_map"][node_id]
    if reference_images is None:
        reference_images = get_reference_batch(cfg["data_root"], cfg["reference_batch_size"])

    logits_i = get_node_logits(node_id, cfg, reference_images, logits_cache)

    # --- Stage 7: get this round's band and cost weight ---
    phase_cfg = phase_control.load_config()
    if T is None:
        T = phase_cfg["T"]  # fallback for standalone calls without an explicit T
    phase_params = phase_control.get_phase_params(round_num, T, phase_cfg)
    d_min, d_max = phase_params["d_min"], phase_params["d_max"]
    w_cost = phase_params["w_cost"]

    max_hops = cfg["hypercube_max_hops"]

    per_neighbor_info = {}
    F_i_t = []

    for peer_id in neighbor_ids:
        hop_distance = (hop_distances or {}).get(peer_id, cfg["hop_distance_default"])

        logits_j = get_node_logits(peer_id, cfg, reference_images, logits_cache)

        factors = compute_cdb_factors(
            logits_i, logits_j, hop_distance=hop_distance, peer_node_id=peer_id
        )
        cost_score = cost_to_cheapness_score(factors["communication_cost"], max_hops)
        utility = compute_utility(factors, cost_score, cfg, w_cost)
        in_band = d_min <= factors["diversity"] <= d_max

        per_neighbor_info[peer_id] = {
            "factors": factors,
            "cost_score": cost_score,
            "utility": utility,
            "in_band": in_band,
        }

        if in_band:
            F_i_t.append(peer_id)

    if not F_i_t:
        return None, F_i_t, per_neighbor_info, phase_params

    j_star = max(F_i_t, key=lambda p: per_neighbor_info[p]["utility"])
    return j_star, F_i_t, per_neighbor_info, phase_params


# ---- Standalone test / inspection -----------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    topology = load_topology()

    test_node_id = os.environ.get("NODE_ID", "node0")
    test_round = int(os.environ.get("ROUND", "1"))

    print(f"Running Stage 5 teacher selection for {test_node_id}, round {test_round}...")

    j_star, F_i_t, info, phase_params = select_teacher(test_node_id, cfg, topology, round_num=test_round)

    print(f"Phase: {phase_params['phase_label']} (t/T={phase_params['round_frac']:.3f})  "
          f"Band: [{phase_params['d_min']:.3f}, {phase_params['d_max']:.3f}]  "
          f"Weights: w_u={cfg['w_u']} w_c={cfg['w_c']} "
          f"w_cost={phase_params['w_cost']:.3f} w_r={cfg['w_r']}\n")

    neighbor_ids = topology["neighbor_map"][test_node_id]
    print(f"N({test_node_id}) = {neighbor_ids}\n")

    for peer_id in neighbor_ids:
        rec = info[peer_id]
        f = rec["factors"]
        band_flag = "IN BAND " if rec["in_band"] else "out of band"
        print(f"  {peer_id}: d_ij={f['diversity']:.4f} [{band_flag}]  "
              f"util={f['utility']:.4f} conf={f['confidence']:.4f} "
              f"cost_score={rec['cost_score']:.4f} rel={f['reliability']:.4f}  "
              f"=> U_i(j)={rec['utility']:.4f}")

    print(f"\nF_i^t (band-filtered candidates) = {F_i_t}")
    if j_star is not None:
        print(f"Selected teacher j* = {j_star}  (highest utility in F_i^t)")
    else:
        print("F_i^t is empty -- no qualifying neighbor. "
              "This client trains on local data only this round (Algorithm 1, line 9).")
