"""
node/train.py

Stage 0: local warm-up training for a single node.

Loads this node's assigned data partition (based on NODE_ID), loads the
model, trains locally for a configurable number of epochs, evaluates
against CIFAR-10's shared test set, and logs per-epoch results to a
per-node CSV under logs/.

Usage (manual test, on host):
    NODE_ID=node0 python node/train.py --config configs/train_config.yaml
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import get_model


def load_node_partition(partitions_dir: str, node_id: str) -> TensorDataset:
    """Load this node's assigned .pt partition file into a TensorDataset."""
    path = os.path.join(partitions_dir, f"{node_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No partition file found for {node_id} at {path}. "
            f"Did you run datasets/partition.py first?"
        )
    data = torch.load(path)
    return TensorDataset(data["images"], data["labels"])


def load_shared_test_set(data_root: str) -> TensorDataset:
    """Load CIFAR-10's own held-out test set - shared across all nodes,
    used purely for evaluation, never for training."""
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return test_set


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / total, loss_sum / total


def init_log_file(log_path: str):
    """Create the per-node CSV with a header row if it doesn't already exist."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "node_id", "stage", "train_loss", "train_acc",
                "test_loss", "test_acc", "train_time_sec"
            ])


def log_round(log_path: str, row: dict):
    """Append one row to the node's CSV log."""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["round"], row["node_id"], row["stage"], row["train_loss"],
            row["train_acc"], row["test_loss"], row["test_acc"], row["train_time_sec"]
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    node_id = os.environ.get("NODE_ID", cfg.get("default_node_id", "node0"))
    device = torch.device("cpu")

    print(f"[{node_id}] Starting Stage 0 warm-up training on device={device}")

    train_set = load_node_partition(cfg["partitions_dir"], node_id)
    test_set = load_shared_test_set(cfg["data_root"])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    print(f"[{node_id}] Local train samples: {len(train_set)}, shared test samples: {len(test_set)}")

    model = get_model(cfg["model_name"], num_classes=cfg["num_classes"], seed=cfg["seed"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    log_path = os.path.join(cfg["log_dir"], f"{node_id}.csv")
    init_log_file(log_path)

    for epoch in range(1, cfg["warmup_epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)

        train_time = time.time() - epoch_start
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        test_acc, test_loss = evaluate(model, test_loader, device)

        print(f"[{node_id}] epoch {epoch}/{cfg['warmup_epochs']} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
              f"time={train_time:.1f}s")

        log_round(log_path, {
            "round": epoch,
            "node_id": node_id,
            "stage": "warmup",
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
            "train_time_sec": round(train_time, 2),
        })

    weights_dir = cfg.get("weights_dir", "logs/weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{node_id}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"[{node_id}] Trained weights saved to {weights_path}")

    print(f"[{node_id}] Warm-up complete. Log written to {log_path}")


if __name__ == "__main__":
    main()
