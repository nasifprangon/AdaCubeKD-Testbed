"""
One-off patch script: threads an explicit T parameter through
select_teacher() and run_distillation_round(), instead of each
independently reloading phase_control_config.yaml (which always used the
file's T=30, ignoring run_experiment.py's --rounds override).

Run with: python patch/apply_patch.py
"""

import re

# ---- Patch node/teacher_selection.py ----------------------------------------

path = "node/teacher_selection.py"
with open(path, "r") as f:
    content = f.read()

old_sig = (
    "def select_teacher(node_id: str, cfg: dict, topology: dict, round_num: int = 1,\n"
    "                    hop_distances: dict = None, logits_cache: dict = None,\n"
    "                    reference_images: torch.Tensor = None):"
)
new_sig = (
    "def select_teacher(node_id: str, cfg: dict, topology: dict, round_num: int = 1,\n"
    "                    hop_distances: dict = None, logits_cache: dict = None,\n"
    "                    reference_images: torch.Tensor = None, T: int = None):"
)
assert old_sig in content, "select_teacher signature not found -- file may already be patched or differs from expected"
content = content.replace(old_sig, new_sig)

old_phase_block = (
    "    # --- Stage 7: get this round's band and cost weight ---\n"
    "    phase_cfg = phase_control.load_config()\n"
    "    T = phase_cfg[\"T\"]\n"
    "    phase_params = phase_control.get_phase_params(round_num, T, phase_cfg)"
)
new_phase_block = (
    "    # --- Stage 7: get this round's band and cost weight ---\n"
    "    phase_cfg = phase_control.load_config()\n"
    "    if T is None:\n"
    "        T = phase_cfg[\"T\"]  # fallback for standalone calls without an explicit T\n"
    "    phase_params = phase_control.get_phase_params(round_num, T, phase_cfg)"
)
assert old_phase_block in content, "phase block not found in teacher_selection.py"
content = content.replace(old_phase_block, new_phase_block)

with open(path, "w") as f:
    f.write(content)
print(f"Patched {path}")

# ---- Patch node/distillation_update.py ---------------------------------------

path = "node/distillation_update.py"
with open(path, "r") as f:
    content = f.read()

old_sig = (
    "def run_distillation_round(node_id: str, cfg: dict, round_num: int = 1,\n"
    "                            logits_cache: dict = None, reference_images: torch.Tensor = None):"
)
new_sig = (
    "def run_distillation_round(node_id: str, cfg: dict, round_num: int = 1,\n"
    "                            logits_cache: dict = None, reference_images: torch.Tensor = None,\n"
    "                            T: int = None):"
)
assert old_sig in content, "run_distillation_round signature not found"
content = content.replace(old_sig, new_sig)

old_call = (
    "    j_star, F_i_t, info, phase_params = select_teacher(\n"
    "        node_id, ts_cfg, topology, round_num=round_num,\n"
    "        logits_cache=logits_cache, reference_images=reference_images,\n"
    "    )"
)
new_call = (
    "    j_star, F_i_t, info, phase_params = select_teacher(\n"
    "        node_id, ts_cfg, topology, round_num=round_num,\n"
    "        logits_cache=logits_cache, reference_images=reference_images, T=T,\n"
    "    )"
)
assert old_call in content, "select_teacher call not found in distillation_update.py"
content = content.replace(old_call, new_call)

with open(path, "w") as f:
    f.write(content)
print(f"Patched {path}")

# ---- Patch controller/run_experiment.py ---------------------------------------

path = "controller/run_experiment.py"
with open(path, "r") as f:
    content = f.read()

old_call = (
    "        j_star, d_ij = run_distillation_round(\n"
    "            node_id, distill_cfg, round_num=round_num,\n"
    "            logits_cache=logits_cache, reference_images=reference_images,\n"
    "        )"
)
new_call = (
    "        j_star, d_ij = run_distillation_round(\n"
    "            node_id, distill_cfg, round_num=round_num,\n"
    "            logits_cache=logits_cache, reference_images=reference_images, T=T,\n"
    "        )"
)
assert old_call in content, "run_distillation_round call not found in run_experiment.py"
content = content.replace(old_call, new_call)

with open(path, "w") as f:
    f.write(content)
print(f"Patched {path}")

print("\nAll three files patched successfully.")
