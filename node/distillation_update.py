"""
node/distillation_update.py

Stage 6: distillation update.

Client i updates its local model using local cross-entropy plus a KL
distillation term toward the Stage-5-selected teacher j*:

    L_i = L_CE(theta_i; xi ~ D_i)
          + lambda_t * tau_t^2 * (1/m) * sum_{x in D_ref}
                KL( p_{j*}^{tau_t}(x) || p_i^{tau_t}(x) )

The tau_t^2 factor is the standard KD gradient-scale correction (Hinton
et al.). If Stage 5 finds no qualifying teacher (F_i^t empty), i trains
on local loss only this round (Algorithm 1, line 9).

STAGE 7 INTEGRATION: lambda_t and tau_t are now obtained from
node/phase_control.py's get_phase_params(round_num, T, phase_cfg), which
implements the paper's phase-aware schedule (Section IV-C). The same
round_num is passed through to Stage 5's select_teacher so both stages
use a consistent phase for a given round.

This module reuses node/train.py's local-training setup (partition
loading, model init, optimizer, CSV logging) and node/teacher_selection.py's
select_teacher for Stage 5, so the same model/data conventions are used
throughout the pipeline.
"""

import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import get_model
from node.train import load_node_partition, load_shared_test_set, evaluate, init_log_file
from node.logit_exchange import get_reference_batch, load_this_node_model
from node.teacher_selection import load_config as load_teacher_selection_config
from node.teacher_selection import load_topology, select_teacher
from node import phase_control


# ---- Config loading -----------------------------------------------------------

def load_config(path: str = "configs/distillation_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---- Temperature-scaled distributions -----------------------------------------
# peer_scoring.py's logits_to_probs() is plain softmax (temperature=1), used
# for Stage 2-3/5 scoring. Stage 6's KD loss needs temperature-scaled
# versions per Eq. (the L_i expression), so those helpers are defined here
# rather than modifying peer_scoring.py.

def temperature_log_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """log p^tau(x) = log_softmax(logits / tau) -- used for the student (input
    side of KL, since torch's kl_div expects log-probabilities for input)."""
    return F.log_softmax(logits / tau, dim=-1)


def temperature_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """p^tau(x) = softmax(logits / tau) -- used for the (frozen) teacher target."""
    return F.softmax(logits / tau, dim=-1)


def kd_kl_loss(student_logits: torch.Tensor, teacher_probs_tau: torch.Tensor,
               tau: float) -> torch.Tensor:
    """
    KL( p_j*^tau || p_i^tau ), averaged over the reference batch, scaled by
    tau^2 (the standard KD gradient-scale correction).

    teacher_probs_tau must already be temperature-scaled softmax probs,
    detached (no gradient through the frozen teacher).
    torch.nn.functional.kl_div(input, target, reduction='batchmean') computes
    mean_x sum_c target(x,c) * (log target(x,c) - input(x,c)), i.e. exactly
    KL(target || softmax(input)) when input is given as log-probabilities --
    which matches KL(p_j* || p_i) with student as the "input" side.
    """
    student_log_probs = temperature_log_softmax(student_logits, tau)
    kl = F.kl_div(student_log_probs, teacher_probs_tau, reduction="batchmean")
    return kl * (tau ** 2)


# ---- CSV logging (extends train.py's schema with distillation info) -----------

def init_distill_log_file(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "node_id", "stage", "phase", "teacher_id", "lambda_t", "tau_t",
                "ce_loss", "kd_loss", "total_loss", "train_acc", "test_loss", "test_acc",
                "train_time_sec"
            ])


def log_distill_round(log_path: str, row: dict):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["round"], row["node_id"], row["stage"], row["phase"], row["teacher_id"],
            row["lambda_t"], row["tau_t"], row["ce_loss"], row["kd_loss"],
            row["total_loss"], row["train_acc"], row["test_loss"], row["test_acc"],
            row["train_time_sec"],
        ])


# ---- Stage 6 core ---------------------------------------------------------------

def run_distillation_round(node_id: str, cfg: dict, round_num: int = 1):
    """
    Run one round of Stage 6 for a single client `node_id`:
      1. Run Stage 5 to get j* (or None), using this round's phase-controlled
         band and cost weight.
      2. Get this round's lambda_t, tau_t from Stage 7.
      3. Train `epochs_per_round` local epochs on D_i, combining CE loss
         with the KD term toward j* (if any) computed on D_ref.
      4. Save updated weights back to logs/weights/{node_id}.pt, and log
         the round to logs/{node_id}.csv (stage="distill").
    """
    device = torch.device("cpu")

    # --- Stage 5: select teacher (band/cost weight from Stage 7 internally) ---
    ts_cfg = load_teacher_selection_config()
    topology = load_topology()
    j_star, F_i_t, info, phase_params = select_teacher(node_id, ts_cfg, topology, round_num=round_num)

    lambda_t = phase_params["lambda_t"]
    tau_t = phase_params["tau_t"]
    phase_label = phase_params["phase_label"]

    print(f"[{node_id}] Stage 5 result: F_i^t={F_i_t}, j*={j_star}")
    print(f"[{node_id}] Stage 7 phase: {phase_label} (t/T={phase_params['round_frac']:.3f})  "
          f"lambda_t={lambda_t:.3f} tau_t={tau_t:.3f}")

    # --- Local data + model (continuing from warmed-up / prior-round weights) ---
    train_set = load_node_partition(cfg["partitions_dir"], node_id)
    test_set = load_shared_test_set(cfg["data_root"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    model = load_this_node_model(cfg, node_id)  # loads logs/weights/{node_id}.pt
    model.to(device)

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    # --- Frozen teacher logits on D_ref (computed once per round; teacher
    # does not update during i's local steps) ---
    reference_images = get_reference_batch(cfg["data_root"], cfg["reference_batch_size"])

    teacher_probs_tau = None
    if j_star is not None:
        teacher_model = load_this_node_model(cfg, j_star)
        teacher_model.to(device)
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = teacher_model(reference_images.to(device))
            teacher_probs_tau = temperature_softmax(teacher_logits, tau_t).detach()

    # --- Training loop ---
    log_path = os.path.join(cfg["log_dir"], f"{node_id}.csv")
    init_distill_log_file(log_path)

    for epoch in range(1, cfg["epochs_per_round"] + 1):
        model.train()
        epoch_start = time.time()
        running_ce, running_kd, running_correct, running_total = 0.0, 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = ce_criterion(outputs, labels)

            if j_star is not None:
                student_ref_logits = model(reference_images.to(device))
                kd_loss = kd_kl_loss(student_ref_logits, teacher_probs_tau, tau_t)
                total_loss = ce_loss + lambda_t * kd_loss
            else:
                kd_loss = torch.tensor(0.0)
                total_loss = ce_loss

            total_loss.backward()
            optimizer.step()

            running_ce += ce_loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)

        train_time = time.time() - epoch_start
        ce_avg = running_ce / running_total
        kd_avg = running_kd / running_total
        total_avg = ce_avg + lambda_t * kd_avg if j_star is not None else ce_avg
        train_acc = running_correct / running_total

        test_acc, test_loss = evaluate(model, test_loader, device)

        print(f"[{node_id}] round {round_num} epoch {epoch}/{cfg['epochs_per_round']} "
              f"phase={phase_label} teacher={j_star} ce={ce_avg:.4f} kd={kd_avg:.4f} "
              f"total={total_avg:.4f} train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f} time={train_time:.1f}s")

        log_distill_round(log_path, {
            "round": round_num,
            "node_id": node_id,
            "stage": "distill",
            "phase": phase_label,
            "teacher_id": j_star if j_star is not None else "none",
            "lambda_t": round(lambda_t, 4),
            "tau_t": round(tau_t, 4),
            "ce_loss": round(ce_avg, 4),
            "kd_loss": round(kd_avg, 4),
            "total_loss": round(total_avg, 4),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
            "train_time_sec": round(train_time, 2),
        })

    weights_dir = cfg.get("weights_dir", "logs/weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{node_id}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"[{node_id}] Round {round_num} weights saved to {weights_path}")

    return j_star


if __name__ == "__main__":
    cfg = load_config()
    test_node_id = os.environ.get("NODE_ID", "node0")
    test_round = int(os.environ.get("ROUND", "1"))

    print(f"Running Stage 6 distillation update for {test_node_id}, round {test_round}...")
    print(f"epochs_per_round={cfg['epochs_per_round']}\n")

    run_distillation_round(test_node_id, cfg, round_num=test_round)
