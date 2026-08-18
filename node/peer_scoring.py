"""
node/peer_scoring.py

Stage 2-3: compute the 5-factor CDB (Critical Disagreement Band) compatibility
score between a node and a candidate peer, from exchanged reference-batch
logits. Matches the definitions in Section 3.3:

  1. Diversity / disagreement - sqrt-JS divergence between i's and j's soft
     predictions on the reference set, averaged over the batch.
  2. Knowledge utility - complementary information j offers i, estimated
     from where j is confident on inputs where i is not.
  3. Prediction confidence - j's own calibration on the reference set
     (negative predictive entropy).
  4. Communication cost - proportional to hypercube hop distance; hardcoded
     as a fixed per-hop constant since all containers share one Docker
     bridge network with effectively uniform latency/bandwidth. Hop
     distance defaults to 1 (direct neighbor) until Stage 4 builds the
     real hypercube and can supply true hop distances.
  5. Reliability - j's historical stability across rounds (participation
     and consistency of past updates). STUBBED: this genuinely requires
     round-level history that doesn't exist until the orchestration loop
     (Stage 8) is built. Returns a fixed default for now - do not treat
     this as a real signal yet.
"""

import torch
import torch.nn.functional as F


# ---- Factor 4: Communication cost (hardcoded) -----------------------------

COST_PER_HOP = 1.0  # arbitrary fixed unit - only relative cost between
                      # candidate peers matters for scoring, not the absolute value


def communication_cost(hop_distance: int = 1) -> float:
    """
    Hardcoded, deterministic communication cost - proportional to hop distance
    in the (eventual) hypercube topology. No live bandwidth/latency measurement,
    since containers sit on the same bridge network with effectively uniform cost.

    hop_distance defaults to 1 until Stage 4 (hypercube construction) exists
    and can supply real hop distances between specific node pairs.
    """
    return hop_distance * COST_PER_HOP


# ---- Factor 5: Reliability (stubbed placeholder) ---------------------------

DEFAULT_RELIABILITY = 1.0  # placeholder: "assume fully reliable" until real
                             # round-history tracking exists


def reliability_score(node_id: str = None) -> float:
    """
    STUB. Real implementation needs round-level history (participation rate,
    consistency of past updates) which does not exist until the controller's
    orchestration loop (Stage 8) is built and tracking rounds over time.

    Currently returns a fixed default for every node - this is NOT a real
    signal yet, just a placeholder so the 5-factor score has a value to
    combine until this gets built properly.
    """
    return DEFAULT_RELIABILITY


# ---- Core distributions / distances -----------------------------------------

def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to probability distributions via softmax."""
    return F.softmax(logits, dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two probability distributions, per row.
    JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), where m = 0.5*(p+q).
    Range: [0, ln(2)].
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def sqrt_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    sqrt(JS divergence) - the Jensen-Shannon DISTANCE. Satisfies the triangle
    inequality (unlike plain JS), making it a proper metric.
    """
    js = js_divergence(p, q)
    return torch.sqrt(torch.clamp(js, min=0.0))


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """H(p) = -sum(p * log(p)), per row. Range: [0, ln(num_classes)]."""
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


# ---- Factor 1: Diversity ----------------------------------------------------

def compute_diversity_score(logits_self: torch.Tensor, logits_peer: torch.Tensor) -> float:
    """
    d_ij = sqrt(JS(p_i, p_j)), averaged over the reference batch.
    Range: [0, sqrt(ln(2))] ~= [0, 0.8326].
    0 = identical predictive behavior; max = maximally different (disjoint).
    """
    probs_self = logits_to_probs(logits_self)
    probs_peer = logits_to_probs(logits_peer)
    per_sample_sqrt_js = sqrt_js_divergence(probs_self, probs_peer)
    return per_sample_sqrt_js.mean().item()


# ---- Factor 3: Prediction confidence ----------------------------------------

def compute_confidence_score(logits: torch.Tensor) -> float:
    """
    Negative predictive entropy on the reference batch, averaged.
    Range: [-ln(num_classes), 0]. 0 = maximally confident (one-hot);
    -ln(K) = least confident (uniform over K classes).
    """
    probs = logits_to_probs(logits)
    neg_entropy = -predictive_entropy(probs)
    return neg_entropy.mean().item()


# ---- Factor 2: Knowledge utility ---------------------------------------------

def compute_knowledge_utility(logits_i: torch.Tensor, logits_j: torch.Tensor) -> float:
    """
    Complementary information j offers i: per-sample confidence(j) minus
    confidence(i), clipped at 0 (relu) so only samples where j is MORE
    confident than i contribute - i.e. "where j knows something i doesn't."
    Averaged over the reference batch.

    Uses the same per-sample negative-entropy confidence definition as
    compute_confidence_score, for consistency between factors 2 and 3.
    """
    probs_i = logits_to_probs(logits_i)
    probs_j = logits_to_probs(logits_j)
    conf_i = -predictive_entropy(probs_i)
    conf_j = -predictive_entropy(probs_j)
    complementary = torch.relu(conf_j - conf_i)
    return complementary.mean().item()


# ---- Combined CDB score ------------------------------------------------------

def compute_cdb_factors(logits_i: torch.Tensor, logits_j: torch.Tensor,
                          hop_distance: int = 1, peer_node_id: str = None) -> dict:
    """
    Compute all 5 CDB factors for candidate peer j, from i's perspective.
    Returns a dict of raw factor values - NOT yet combined into a single
    score, since the combination weighting (Section 3.3's actual CDB
    filter logic) hasn't been implemented yet. This is scoped to Stage
    2-3 (feature computation); the filter/combination step is Stage 5.
    """
    return {
        "diversity": compute_diversity_score(logits_i, logits_j),
        "utility": compute_knowledge_utility(logits_i, logits_j),
        "confidence": compute_confidence_score(logits_j),
        "communication_cost": communication_cost(hop_distance),
        "reliability": reliability_score(peer_node_id),
    }


if __name__ == "__main__":
    print("Running self-tests for peer_scoring.py...\n")

    logits_a = torch.randn(20, 10)
    diversity_same = compute_diversity_score(logits_a, logits_a.clone())
    print(f"Diversity (identical logits): {diversity_same:.6f} (expect ~0.0)")
    assert diversity_same < 1e-4

    torch.manual_seed(0)
    logits_b = torch.randn(20, 10)
    logits_c = torch.randn(20, 10) * 5 + 10
    diversity_diff = compute_diversity_score(logits_b, logits_c)
    print(f"Diversity (very different logits): {diversity_diff:.6f} (expect > 0)")
    assert diversity_diff > 0.01

    near_one_hot = torch.tensor([[20.0, -20.0, -20.0, -20.0]])
    conf_confident = compute_confidence_score(near_one_hot)
    print(f"Confidence (near one-hot logits): {conf_confident:.6f} (expect ~0.0, the max)")
    assert conf_confident > -1e-3

    uniform_logits = torch.zeros(1, 4)
    conf_uniform = compute_confidence_score(uniform_logits)
    import math
    expected_min = -math.log(4)
    print(f"Confidence (uniform, 4 classes): {conf_uniform:.6f} (expect {expected_min:.6f})")
    assert abs(conf_uniform - expected_min) < 1e-4

    i_uncertain = torch.randn(50, 10) * 0.1
    j_confident = torch.eye(10)[torch.randint(0, 10, (50,))] * 20 - 10
    u_high = compute_knowledge_utility(i_uncertain, j_confident)
    print(f"Utility (j confident, i uncertain): {u_high:.6f} (expect > 0)")
    assert u_high > 0.1

    u_same = compute_knowledge_utility(i_uncertain, i_uncertain.clone())
    print(f"Utility (identical inputs): {u_same:.6f} (expect ~0)")
    assert abs(u_same) < 1e-4

    u_reverse = compute_knowledge_utility(j_confident, i_uncertain)
    print(f"Utility (reversed - i confident, j not): {u_reverse:.6f} (expect ~0, relu clips)")
    assert u_reverse < 0.01

    print(f"Communication cost (hop=1): {communication_cost(1)} (expect 1.0)")
    print(f"Communication cost (hop=3): {communication_cost(3)} (expect 3.0)")
    assert communication_cost(1) == 1.0
    assert communication_cost(3) == 3.0

    print(f"Reliability (stub, any node): {reliability_score('node3')} (expect 1.0, placeholder)")
    assert reliability_score("node3") == 1.0

    factors = compute_cdb_factors(logits_b, logits_c, hop_distance=2, peer_node_id="node3")
    print(f"\nCombined CDB factors dict: {factors}")
    assert set(factors.keys()) == {"diversity", "utility", "confidence", "communication_cost", "reliability"}

    print("\nAll self-tests passed.")
