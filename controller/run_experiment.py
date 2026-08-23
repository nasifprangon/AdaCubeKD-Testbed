"""
controller/run_experiment.py

Orchestration loop: Algorithm 1's "while" loop (lines 3-13), tying
Stages 5-8 together into an actual multi-round run across all 8 nodes.

Assumes Setup (Algorithm 1, line 1 / Stages 0-4) has already been run
manually: logs/weights/*.pt exist (Stage 0 warm-up) and
results/topology.json exists (Stage 4 embedding). This script does NOT
re-run warm-up training or the initial hypercube construction.

For each round t = 1..T:
  1. Build a FRESH, EMPTY logits_cache at the start of the round (before
     any node has trained this round). Every node's Stage 5 call this
     round shares this one cache, so all nodes score against a consistent
     round-start snapshot of all weights -- matching Algorithm 1's
     "for each client i (parallel) do" semantics -- rather than a
     sequential loop where an earlier node's in-round update leaks into a
     later node's neighbor lookup. This also eliminates redundant disk
     loads when nodes share neighbors. See teacher_selection.py's module
     docstring for the full rationale.
  2. For each node (sequentially -- host-side testing before container
     orchestration, per the project's established workflow): run Stage 6
     (which internally calls Stage 5 for that node), reusing the round's
     shared cache. Stage 6 now returns (j_star, d_ij_to_teacher) directly,
     so no separate Stage-5 call is needed here just to log D_t.
  3. After all nodes finish the round: if Stage 8's should_refresh(t, R)
     is true, run refresh_topology(t, cfg).
  4. Compute and log D_t: the network disagreement statistic (Theorem 1)
     -- the average d_ij between each node and its selected teacher this
     round, over nodes that had a non-empty F_i^t. Nodes with no teacher
     this round are excluded from the D_t average.

IMPORTANT SCOPE NOTE: Algorithm 1 stops "while D_t above noise floor" --
but we do not yet have a validated noise-floor estimate for this testbed.
This script instead runs the full fixed T rounds from
configs/phase_control_config.yaml, and logs D_t every round so a real
noise floor / stopping point can be identified from the data afterward.

Usage (full run):
    python controller/run_experiment.py

Usage (smoke test -- override rounds/nodes for a quick sanity check):
    python controller/run_experiment.py --rounds 3 --nodes node0,node1
"""

import argparse
import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node.distillation_update import load_config as load_distill_config
from node.distillation_update import run_distillation_round
from node.logit_exchange import get_reference_batch
from node import phase_control
from controller.periodic_refresh import (
    load_config as load_refresh_config,
    should_refresh,
    refresh_topology,
)


ALL_NODE_IDS = [f"node{i}" for i in range(8)]


def init_experiment_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "phase", "d_min", "d_max", "lambda_t", "tau_t", "w_cost",
                "num_nodes_with_teacher", "num_nodes_total", "D_t",
                "refreshed", "rewired", "drift", "round_time_sec",
            ])


def log_experiment_round(log_path: str, row: dict):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["round"], row["phase"], row["d_min"], row["d_max"], row["lambda_t"],
            row["tau_t"], row["w_cost"], row["num_nodes_with_teacher"],
            row["num_nodes_total"], row["D_t"], row["refreshed"], row["rewired"],
            row["drift"], row["round_time_sec"],
        ])


def run_round(round_num: int, node_ids: list, distill_cfg: dict, phase_cfg: dict,
              T: int, reference_images):
    """
    Run Stage 6 (which internally runs Stage 5) for every node this round,
    all sharing one round-start logits_cache. Returns (D_t,
    num_with_teacher, phase_params).
    """
    phase_params = phase_control.get_phase_params(round_num, T, phase_cfg)
    logits_cache = {}  # fresh each round -- round-start snapshot only
    d_ij_with_teacher = []

    for node_id in node_ids:
        j_star, d_ij = run_distillation_round(
            node_id, distill_cfg, round_num=round_num,
            logits_cache=logits_cache, reference_images=reference_images, T=T,
        )
        if d_ij is not None:
            d_ij_with_teacher.append(d_ij)

    D_t = sum(d_ij_with_teacher) / len(d_ij_with_teacher) if d_ij_with_teacher else None
    return D_t, len(d_ij_with_teacher), phase_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None,
                         help="Override T (total rounds) for a smoke test. "
                              "Default: use T from configs/phase_control_config.yaml.")
    parser.add_argument("--nodes", type=str, default=None,
                         help="Comma-separated node IDs to run, e.g. node0,node1. "
                              "Default: all 8 nodes.")
    parser.add_argument("--log_path", type=str, default="logs/experiment.csv")
    args = parser.parse_args()

    distill_cfg = load_distill_config()
    phase_cfg = phase_control.load_config()
    refresh_cfg = load_refresh_config()

    T = args.rounds if args.rounds is not None else phase_cfg["T"]
    node_ids = args.nodes.split(",") if args.nodes else ALL_NODE_IDS
    R = refresh_cfg["R"]

    # Reference batch is deterministic (fixed first N CIFAR-10 test images),
    # so it's safe and slightly faster to build it once for the whole run
    # rather than regenerating it every round.
    reference_images = get_reference_batch(distill_cfg["data_root"], distill_cfg["reference_batch_size"])

    print(f"Starting AdaCubeKD orchestration loop: T={T} rounds, "
          f"{len(node_ids)} nodes ({node_ids}), refresh period R={R}")
    if args.rounds is not None or args.nodes is not None:
        print("NOTE: running with --rounds/--nodes override (smoke test), "
              "not the full configured experiment.\n")

    init_experiment_log(args.log_path)

    for t in range(1, T + 1):
        round_start = time.time()

        D_t, num_with_teacher, phase_params = run_round(
            t, node_ids, distill_cfg, phase_cfg, T, reference_images
        )

        refreshed, rewired, drift = False, False, None
        if should_refresh(t, R):
            result = refresh_topology(t, refresh_cfg)
            refreshed = True
            rewired = result["rewired"]
            drift = result["drift"]

        round_time = time.time() - round_start

        D_t_display = f"{D_t:.4f}" if D_t is not None else "N/A (no active edges)"
        print(f"\n=== Round {t}/{T} complete === phase={phase_params['phase_label']} "
              f"D_t={D_t_display}  nodes_with_teacher={num_with_teacher}/{len(node_ids)}  "
              f"refreshed={refreshed}" + (f" rewired={rewired} drift={drift:.4f}" if refreshed else "") +
              f"  round_time={round_time:.1f}s\n")

        log_experiment_round(args.log_path, {
            "round": t,
            "phase": phase_params["phase_label"],
            "d_min": round(phase_params["d_min"], 4),
            "d_max": round(phase_params["d_max"], 4),
            "lambda_t": round(phase_params["lambda_t"], 4),
            "tau_t": round(phase_params["tau_t"], 4),
            "w_cost": round(phase_params["w_cost"], 4),
            "num_nodes_with_teacher": num_with_teacher,
            "num_nodes_total": len(node_ids),
            "D_t": round(D_t, 4) if D_t is not None else "",
            "refreshed": refreshed,
            "rewired": rewired,
            "drift": round(drift, 4) if drift is not None else "",
            "round_time_sec": round(round_time, 2),
        })

    print(f"Experiment complete. {T} rounds logged to {args.log_path}")


if __name__ == "__main__":
    main()
