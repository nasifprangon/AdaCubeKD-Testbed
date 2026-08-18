"""
node/logit_exchange.py

Stage 1: generate logits on a shared reference batch, then exchange them
with a peer node over TCP.

Every node evaluates its (locally trained) model on the SAME fixed
reference batch - the first N images from CIFAR-10's test set, same for
every node - so the resulting logits are directly comparable across
nodes. This is what Stage 2's peer scoring (sqrt-JS divergence) will
consume.

Since TCP is a byte stream (not message-based), each logits tensor is
sent length-prefixed: a 4-byte big-endian length header, followed by
the serialized tensor payload.

Usage - LISTENER (waits for one peer to connect and send logits):
    NODE_ID=node3 python node/logit_exchange.py --mode listen --port 9999

Usage - SENDER (connects to a peer and sends its own logits):
    NODE_ID=node0 python node/logit_exchange.py --mode send --peer_host node3 --port 9999
"""

import argparse
import io
import os
import socket
import struct
import sys

import torch
import yaml
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import get_model


def get_reference_batch(data_root: str, num_samples: int):
    """
    Load the first `num_samples` images from CIFAR-10's test set - fixed
    and identical across all nodes, so every node's logits are computed
    on exactly the same inputs and are therefore comparable.
    """
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    images = torch.stack([test_set[i][0] for i in range(num_samples)])
    return images


def compute_logits(model: torch.nn.Module, reference_images: torch.Tensor) -> torch.Tensor:
    """Run the reference batch through the model, returning raw (pre-softmax) logits."""
    model.eval()
    with torch.no_grad():
        logits = model(reference_images)
    return logits


def send_logits(sock: socket.socket, logits: torch.Tensor):
    """Serialize logits and send them length-prefixed over the socket."""
    buf = io.BytesIO()
    torch.save(logits, buf)
    payload = buf.getvalue()
    sock.sendall(struct.pack(">I", len(payload)) + payload)
    return len(payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from the socket, or raise if the connection closes early."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before expected data was fully received")
        data += chunk
    return data


def recv_logits(sock: socket.socket) -> torch.Tensor:
    """Receive a length-prefixed logits tensor from the socket."""
    length_bytes = recv_exact(sock, 4)
    length = struct.unpack(">I", length_bytes)[0]
    payload = recv_exact(sock, length)
    buf = io.BytesIO(payload)
    return torch.load(buf)


def load_this_node_model(cfg: dict, node_id: str) -> torch.nn.Module:
    """
    Load this node's model with its WARMED-UP weights (from Stage 0),
    not a fresh random init - logit exchange should reflect what the
    node actually learned locally.
    """
    model = get_model(cfg["model_name"], num_classes=cfg["num_classes"], seed=cfg["seed"])
    weights_path = os.path.join(cfg.get("weights_dir", "logs/weights"), f"{node_id}.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"[{node_id}] Loaded warmed-up weights from {weights_path}")
    else:
        print(f"[{node_id}] WARNING: no saved weights found at {weights_path}, "
              f"using freshly initialized (untrained) model instead.")
    return model


def run_listener(cfg: dict, node_id: str, port: int):
    """Listen for one incoming connection, receive logits, print a summary."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    print(f"[{node_id}] Listening on port {port} for incoming logits...")

    conn, addr = s.accept()
    print(f"[{node_id}] Connection accepted from {addr}")
    received_logits = recv_logits(conn)
    conn.close()
    s.close()

    print(f"[{node_id}] Received logits, shape={tuple(received_logits.shape)}")
    print(f"[{node_id}] Sample of received logits (first row): {received_logits[0].tolist()}")
    return received_logits


def run_sender(cfg: dict, node_id: str, peer_host: str, port: int):
    """Compute this node's own logits on the reference batch and send them to a peer."""
    reference_images = get_reference_batch(cfg["data_root"], cfg["reference_batch_size"])
    model = load_this_node_model(cfg, node_id)
    my_logits = compute_logits(model, reference_images)

    print(f"[{node_id}] Computed logits on reference batch, shape={tuple(my_logits.shape)}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[{node_id}] Connecting to {peer_host}:{port} ...")
    s.connect((peer_host, port))
    bytes_sent = send_logits(s, my_logits)
    s.close()

    print(f"[{node_id}] Sent {bytes_sent} bytes of logits to {peer_host}:{port}")
    return my_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/logit_exchange_config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["listen", "send"])
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--peer_host", type=str, default=None,
                         help="Required when --mode send: hostname of the peer to connect to.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    node_id = os.environ.get("NODE_ID", cfg.get("default_node_id", "node0"))

    if args.mode == "listen":
        run_listener(cfg, node_id, args.port)
    elif args.mode == "send":
        if not args.peer_host:
            raise ValueError("--peer_host is required when --mode send")
        run_sender(cfg, node_id, args.peer_host, args.port)


if __name__ == "__main__":
    main()
