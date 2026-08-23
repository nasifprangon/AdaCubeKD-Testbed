# AdaCubeKD-Testbed

A decentralized federated knowledge-distillation testbed implementing **AdaCubeKD**: 8 simulated edge nodes on a logical hypercube topology, mixing knowledge via adaptive peer selection (Critical Disagreement Band) and knowledge distillation instead of direct weight averaging.

> Paper: *AdaCubeKD: Utility-Driven Adaptive Peer Selection for Structured Federated Distillation* — Nasif Fahmid Prangon, Jie Wu (Temple University)

## Status

All 9 pipeline stages are implemented and individually verified. The orchestration loop tying them together has been smoke-tested at small scale (2 nodes/3 rounds, 4 nodes/5 rounds). A full 8-node, 30-round run has not yet been executed. See [Next Steps](#next-steps) for what's outstanding.

| Stage | Name | Status |
|---|---|---|
| 0 | Local warm-up | ✅ Done |
| 1 | Logit exchange | ✅ Done |
| 2–3 | Peer scoring (5-factor CDB) | ✅ Done (reliability stubbed) |
| 4 | Hypercube construction | ✅ Done |
| 5 | Teacher selection (CDB filter) | ✅ Done |
| 6 | Distillation update | ✅ Done |
| 7 | Phase-aware control | ✅ Done (untuned schedule) |
| 8 | Periodic refresh | ✅ Done (reliability tracking deferred) |
| — | Orchestration loop | ✅ Done, smoke-tested only |

## Environment

- **Host:** Windows + WSL2, Ubuntu 24.04.1 LTS
- **Python:** 3.12, isolated via `venv/`
- **Compute:** CPU-only by design (`device: cpu` in every config) — a GT 1030's 2GB VRAM can't support 8 concurrent training containers, but configs are structured so a real GPU machine is a one-line swap
- **Docker:** Engine + Compose via the official apt repo (not snap, not `docker.io`)

```bash
# Resume a session
sudo service docker start
cd ~/AdaCubeKD_Testbed
source venv/bin/activate

# Install (host)
pip install -r requirements.txt
```

## Repository Layout

```
controller/   orchestrator: hypercube construction, periodic refresh,
              the multi-round experiment loop. Never trains a model.
node/         code every node runs; identity comes from NODE_ID env var.
models/       shared model definitions (SimpleCNN), selected via config.
datasets/     CIFAR-10 Dirichlet-alpha partitioning logic.
configs/      one YAML per pipeline stage; version-controlled.
logs/         per-node CSV metrics + saved weights. Gitignored.
results/      controller-aggregated outputs (topology.json). Gitignored.
analysis/     notebooks/scripts for turning results/ into figures.
docker/       shared Dockerfile for controller + node roles.
patch/        one-off Python patch scripts for surgical bug fixes.
```

Development follows a **test-on-host-first** workflow: every stage is verified with `NODE_ID=nodeX python node/stage_script.py`-style manual invocations before any container orchestration.

## Quick Start

Assumes Stages 0–4 (warm-up training + initial hypercube) have already been run and `logs/weights/*.pt` + `results/topology.json` exist.

```bash
# Smoke test: 2 nodes, 3 rounds (fast, for sanity-checking changes)
python controller/run_experiment.py --rounds 3 --nodes node0,node1

# Full run: all 8 nodes, T from configs/phase_control_config.yaml
python controller/run_experiment.py
```

Results are logged to `logs/experiment.csv` (one row per round: phase, band, λ_t, τ_t, D_t, refresh/rewire events).

### Testing an individual stage

```bash
NODE_ID=node0 python node/train.py                     # Stage 0
NODE_ID=node0 ROUND=1 python node/teacher_selection.py  # Stage 5
NODE_ID=node0 ROUND=1 python node/distillation_update.py # Stage 6
python node/phase_control.py                            # Stage 7 schedule table
ROUND=5 python controller/periodic_refresh.py            # Stage 8
```

## Pipeline Summary

**Setup (Stages 0–4, run once):** local warm-up training → broadcast logits on a shared reference batch → compute the 5-factor compatibility score between every pair (√JS diversity, knowledge utility, confidence, communication cost, reliability) → embed all 8 clients into a hypercube using a spread-preserving, diversity-only construction (each client gets one low-, one moderate-, one high-disagreement neighbor).

**Training loop (Stages 5–8, every round):**
- **Stage 5** filters each client's fixed neighbors to the *Critical Disagreement Band* `[d_min^t, d_max^t]`, then picks the highest-utility survivor as teacher (or trains locally only, if none qualify).
- **Stage 6** trains on local cross-entropy + a temperature-scaled KL distillation term toward the selected teacher.
- **Stage 7** adapts the band, distillation strength `λ_t`, temperature `τ_t`, and cost weight `w_cost^t` across three qualitative phases (Early = stability, Middle = diversity, Late = efficiency), via smoothstep interpolation between phase plateau values.
- **Stage 8** re-scores peers every `R` rounds using current (not frozen) weights, and rewires the hypercube if compatibility drift exceeds a threshold.

## Known Design Placeholders

Several constants are **documented, deliberate placeholders** — functionally complete but not derived from the paper's math or tuned against real observed behavior on this testbed (flagged in each config file's comments):

| Quantity | Current value | Owned by |
|---|---|---|
| Reliability score | constant `1.0` | `node/peer_scoring.py` (stub) |
| CDB band per phase | Early `[.30,.45]` / Mid `[.25,.60]` / Late `[.35,.50]` | `configs/phase_control_config.yaml` |
| `λ_t`, `τ_t` per phase | `.8/.5/.2`, `4/2/1` | same |
| `w_cost^t` per phase | `.05/.10/.30` | same |
| `T`, phase boundaries | `T=30`, equal thirds | same |
| Refresh period `R` | `5` | `configs/periodic_refresh_config.yaml` |
| Drift threshold | `0.05` | same |

These should be recalibrated against real `D_t`/drift data once a full multi-round run has been executed.

## Next Steps

- [ ] Full 8-node, `T=30` unattended run (~1–1.3 hrs estimated on CPU)
- [ ] Reset `logs/weights/` to a clean, uniform Stage-0 state before that run (ad hoc testing has left nodes at uneven training progress)
- [ ] Regenerate `results/topology.json` from a clean Stage 4 run if a fresh starting topology is desired
- [ ] Real reliability tracking (Stage 8 follow-up), replacing the constant stub
- [ ] Calibrate placeholder constants against real observed data
- [ ] Dirichlet-α heterogeneity sweep (`α ∈ {0.1, 0.5, 1.0}`)
- [ ] Full container orchestration pass (currently host-only)
- [ ] Baselines: DFedAvg, D-PSGD, DFML

## Notes on Verified Correctness

A few things worth knowing if you're reading the code:

- **`d_ij` is always recomputed live** during Stages 5 and 8, not read from the frozen setup-time snapshot — the snapshot in `topology.json` only ever seeds the fixed hypercube (Stage 4), never drives round-by-round decisions.
- **A round-scoped logits cache** (built fresh each round, shared across all 8 nodes' Stage 5/6 calls) ensures every node scores against a consistent round-start snapshot of weights, matching the paper's "parallel" semantics rather than letting an earlier node's in-round update leak into a later node's neighbor lookup.
- A real bug was caught and fixed where `Stage 5` independently reloaded the phase-control config and ignored an orchestration-loop `--rounds` override, causing logged phase parameters to disagree with what was actually applied during training. See `patch/apply_patch.py` and commit `80e8d61`.

## License

_(add if applicable)_
