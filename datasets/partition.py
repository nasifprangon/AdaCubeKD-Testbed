"""
datasets/partition.py

Stage 0 (prerequisite): partitions CIFAR-10 across N nodes using a
Dirichlet(alpha) split per class, and saves one .pt file per node under
datasets/partitions/alpha_<value>/. This runs ONCE on the host (not inside
containers) - containers just load their assigned file at startup based on
NODE_ID.

Config values (configs/partition_config.yaml) are defaults; any CLI flag
passed in overrides the matching config value for that run, without editing
the file. This makes running the heterogeneity sweep (multiple alpha values)
straightforward:

    python datasets/partition.py --alpha 0.1
    python datasets/partition.py --alpha 0.5
    python datasets/partition.py --alpha 1.0
    python datasets/partition.py --alpha 10.0

Each run writes to its own datasets/partitions/alpha_<value>/ folder, so
sweeping alpha never overwrites a previous split - all of them stay on disk
side by side for later comparison.
"""

import argparse
import os
import yaml
import numpy as np
import torch
from torchvision import datasets, transforms


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int):
    """
    Split sample indices across num_clients using a Dirichlet(alpha) distribution
    per class.

    Low alpha (e.g. 0.1)  -> highly non-IID: each client dominated by a few classes.
    High alpha (e.g. 100) -> close to IID: each client sees ~equal class proportions.

    Returns: list of length num_clients, each a list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # shuffle each client's own index list so classes aren't grouped in order
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


def format_alpha_for_path(alpha: float) -> str:
    """Turn e.g. 0.1 -> '0.1', 10.0 -> '10.0' into a clean folder-name-safe string."""
    return str(alpha).rstrip("0").rstrip(".") if "." in str(alpha) else str(alpha)


def main():
    parser = argparse.ArgumentParser(description="Partition CIFAR-10 across nodes (Dirichlet split).")
    parser.add_argument("--config", type=str, default="configs/partition_config.yaml",
                         help="Path to base config YAML.")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Dirichlet concentration. Overrides config value. "
                              "Lower = more non-IID (e.g. 0.1), higher = closer to IID (e.g. 100).")
    parser.add_argument("--num_nodes", type=int, default=None,
                         help="Number of nodes to split across. Overrides config value.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for the split. Overrides config value.")
    parser.add_argument("--data_root", type=str, default=None,
                         help="Where CIFAR-10 is downloaded/cached. Overrides config value.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Base output directory. Overrides config value. "
                              "A per-alpha subfolder is created inside it automatically.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI args override config values only when explicitly passed (not None)
    num_nodes = args.num_nodes if args.num_nodes is not None else cfg["num_nodes"]
    alpha = args.alpha if args.alpha is not None else cfg["alpha"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_root = args.data_root if args.data_root is not None else cfg["data_root"]
    base_output_dir = args.output_dir if args.output_dir is not None else cfg["output_dir"]

    # tag the output folder with alpha so sweeps don't overwrite each other
    alpha_tag = format_alpha_for_path(alpha)
    output_dir = os.path.join(base_output_dir, f"alpha_{alpha_tag}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    print(f"Config: num_nodes={num_nodes}, alpha={alpha}, seed={seed}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    print(f"\nDownloading/loading CIFAR-10 into {data_root} ...")
    # Deliberately no normalization/augmentation baked in here - partitioning
    # only decides WHICH samples go WHERE. Actual model-facing transforms
    # (normalize, augment) belong in the training loop's dataset loader, not here.
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    labels = np.array(train_set.targets)
    print(f"Total training samples: {len(labels)}, classes: {len(np.unique(labels))}")
    print(f"Partitioning across {num_nodes} nodes with alpha={alpha}, seed={seed} ...\n")

    client_indices = dirichlet_partition(labels, num_clients=num_nodes, alpha=alpha, seed=seed)

    for node_id, indices in enumerate(client_indices):
        images = torch.stack([train_set[i][0] for i in indices])
        node_labels = torch.tensor([train_set[i][1] for i in indices], dtype=torch.long)

        class_counts = np.bincount(node_labels.numpy(), minlength=10)
        out_path = os.path.join(output_dir, f"node{node_id}.pt")
        torch.save({"images": images, "labels": node_labels}, out_path)

        print(f"  node{node_id}: {len(indices)} samples -> {out_path}")
        print(f"    class distribution: {class_counts.tolist()}")

    print(f"\nPartitioning complete. alpha={alpha} split saved under: {output_dir}")


if __name__ == "__main__":
    main()
