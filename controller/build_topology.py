"""
controller/build_topology.py

Stage 4 driver: computes real pairwise diversity (sqrt-JS) scores from all
8 nodes' warmed-up models on the shared reference batch, then builds the
actual hypercube topology and neighbor map.

Runs on the HOST directly against saved weights - no live TCP transfer
needed here, since Stage 1 already proved logits transfer over the real
Docker network byte-for-byte identically to loading them locally. Loading
weights and computing logits directly produces the exact same values a
socket exchange would.
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import get_model
from node.peer_scoring import compute_diversity_score
from controller.hypercube import build_hypercube_recursive, hypercube_edges, get_neighbor_map


def get_reference_batch(data_root: str, num_samples: int) -> torch.Tensor:
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return torch.stack([test_set[i][0] for i in range(num_samples)])


def load_node_model(cfg: dict, node_id: str) -> torch.nn.Module:
    model = get_model(cfg["model_name"], num_classes=cfg["num_classes"], seed=cfg["seed"])
    weights_path = os.path.join(cfg.get("weights_dir", "logs/weights"), f"{node_id}.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No weights found for {node_id} at {weights_path}. "
            f"Run node/train.py for this node first (NODE_ID={node_id})."
        )
    model.load_state_dict(torch.load(weights_path))
    return model


def compute_logits(model: torch.nn.Module, reference_images: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(reference_images)


def main():
    with open("configs/logit_exchange_config.yaml") as f:
        cfg = yaml.safe_load(f)

    n_nodes = 8
    node_names = [f"node{i}" for i in range(n_nodes)]

    print("Loading reference batch...")
    reference_images = get_reference_batch(cfg["data_root"], cfg["reference_batch_size"])

    print("Computing logits for all 8 nodes from their warmed-up weights...")
    all_logits = {}
    for name in node_names:
        model = load_node_model(cfg, name)
        all_logits[name] = compute_logits(model, reference_images)
        print(f"  {name}: logits computed, shape={tuple(all_logits[name].shape)}")

    print(f"\nComputing pairwise diversity matrix ({n_nodes * (n_nodes - 1) // 2} unique pairs)...")
    d_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            div = compute_diversity_score(all_logits[node_names[i]], all_logits[node_names[j]])
            d_matrix[i, j] = div
            d_matrix[j, i] = div

    print("\nDiversity matrix (real, from actual trained nodes):")
    print(np.round(d_matrix, 4))

    print("\nBuilding hypercube topology from real diversity scores...")
    final_order = build_hypercube_recursive(d_matrix)
    edges = hypercube_edges(3)
    neighbor_map = get_neighbor_map(final_order, edges, node_names=node_names)

    print("\n" + "=" * 50)
    print("REAL HYPERCUBE EDGES (from your actual trained nodes)")
    print("=" * 50)
    vertex_to_name = {v: node_names[c] for v, c in enumerate(final_order)}
    edge_rows = []
    for u, v in edges:
        nu, nv = vertex_to_name[u], vertex_to_name[v]
        ci, cj = node_names.index(nu), node_names.index(nv)
        edge_rows.append((nu, nv, d_matrix[ci, cj]))
    edge_rows.sort(key=lambda x: x[2])
    for nu, nv, dist in edge_rows:
        print(f"  {nu:<6} -- {nv:<6}  distance = {dist:.4f}")

    print("\n" + "=" * 50)
    print("NEIGHBOR MAP")
    print("=" * 50)
    for node in node_names:
        print(f"  {node}: {neighbor_map[node]}")

    os.makedirs("results", exist_ok=True)
    output = {
        "diversity_matrix": d_matrix.tolist(),
        "node_names": node_names,
        "neighbor_map": neighbor_map,
        "final_order": list(final_order),
    }
    with open("results/topology.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTopology saved to results/topology.json")


if __name__ == "__main__":
    main()
