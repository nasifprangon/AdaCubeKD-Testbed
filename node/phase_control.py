"""
node/phase_control.py

Stage 7: phase-aware control.

After every round, the phase controller updates the five phase parameters
(d_min^t, d_max^t, lambda_t, tau_t, w_cost^t) for the next round (Section
IV-C). AdaCubeKD moves through three qualitative phases:

  Early (stability):  band narrow, low-to-moderate; lambda_t high; tau_t
                       high (soft targets); w_cost^t low.
  Middle (diversity):  band widens; lambda_t moderates; tau_t anneals down;
                       w_cost^t stays low-to-moderate.
  Late (efficiency):   band tightens again; lambda_t and tau_t both low;
                       w_cost^t rises (cost penalized more heavily).

IMPORTANT SCOPE NOTE: the paper (Section IV-C) defines the phase behavior
QUALITATIVELY only -- no closed-form schedule is given anywhere in the
draft. This module is a concrete instantiation of that qualitative spec
(see the "Concrete instantiation used in our testbed" paragraph, Section
IV-C), NOT a value derived from the paper's math. Phase plateau values,
boundary fractions, and transition width are config-driven placeholders,
not tuned or validated. This is a design choice, documented as such, in
the same spirit as the reliability stub in peer_scoring.py.

Interpolation shape: smoothstep (Hermite, 3x^2 - 2x^3), blended around
each phase boundary within a configurable transition width, so parameters
move continuously rather than jumping discontinuously at round T/3, 2T/3.
"""

import os

import yaml


# ---- Config loading -----------------------------------------------------------

def load_config(path: str = "configs/phase_control_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---- Smoothstep interpolation --------------------------------------------------

def smoothstep(x: float) -> float:
    """Hermite smoothstep: 0 at x<=0, 1 at x>=1, smooth S-curve between."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _blend_weight(round_frac: float, boundary: float, width: float) -> float:
    """
    Returns a smoothstep blend weight in [0,1] centered on `boundary`:
    0 well before it, 1 well after it, smooth transition of total width
    `width` (half on each side of the boundary).
    """
    if width <= 0:
        return 1.0 if round_frac >= boundary else 0.0
    lo = boundary - width / 2.0
    hi = boundary + width / 2.0
    return smoothstep((round_frac - lo) / (hi - lo))


def interpolate_param(round_frac: float, early_val: float, mid_val: float,
                       late_val: float, cfg: dict) -> float:
    """
    Blend early_val -> mid_val -> late_val as a function of round_frac
    (= t/T, in [0,1]), using two smoothstep transitions centered at the
    Early/Middle and Middle/Late boundaries.
    """
    b1 = cfg["boundary1_frac"]   # Early/Middle boundary, default 1/3
    b2 = cfg["boundary2_frac"]   # Middle/Late boundary, default 2/3
    width = cfg["transition_width_frac"]

    w1 = _blend_weight(round_frac, b1, width)  # 0=Early, 1=at-or-past Middle start
    value_after_first_blend = early_val + w1 * (mid_val - early_val)

    w2 = _blend_weight(round_frac, b2, width)  # 0=Middle, 1=at-or-past Late start
    value = value_after_first_blend + w2 * (late_val - value_after_first_blend)

    return value


# ---- Stage 7 core: get_phase_params --------------------------------------------

def get_phase_params(round_num: int, T: int, cfg: dict) -> dict:
    """
    Return the five phase-controlled parameters for round `round_num` out
    of a total of `T` rounds:
        d_min^t, d_max^t, lambda_t, tau_t, w_cost^t

    This is the single function Stage 5 (teacher_selection.py) and Stage 6
    (distillation_update.py) should call each round instead of reading
    flat constants from their own config files.
    """
    round_frac = round_num / T if T > 0 else 0.0

    params = {}
    for key in ["d_min", "d_max", "lambda_t", "tau_t", "w_cost"]:
        early = cfg[f"{key}_early"]
        mid = cfg[f"{key}_mid"]
        late = cfg[f"{key}_late"]
        params[key] = interpolate_param(round_frac, early, mid, late, cfg)

    params["round_frac"] = round_frac
    params["phase_label"] = phase_label(round_frac, cfg)
    return params


def phase_label(round_frac: float, cfg: dict) -> str:
    """Qualitative phase label for a given round fraction, for logging/printing."""
    if round_frac < cfg["boundary1_frac"]:
        return "Early"
    elif round_frac < cfg["boundary2_frac"]:
        return "Middle"
    else:
        return "Late"


# ---- Standalone test / inspection -----------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    T = cfg["T"]

    print(f"Stage 7 phase schedule, T={T} rounds, "
          f"boundaries at t/T={cfg['boundary1_frac']:.3f} / {cfg['boundary2_frac']:.3f}, "
          f"transition_width_frac={cfg['transition_width_frac']:.3f}\n")

    header = f"{'t':>3} {'phase':>7} {'frac':>6} {'d_min':>7} {'d_max':>7} " \
             f"{'lambda_t':>9} {'tau_t':>7} {'w_cost':>7}"
    print(header)
    print("-" * len(header))

    for t in range(1, T + 1):
        p = get_phase_params(t, T, cfg)
        print(f"{t:>3} {p['phase_label']:>7} {p['round_frac']:>6.3f} "
              f"{p['d_min']:>7.3f} {p['d_max']:>7.3f} "
              f"{p['lambda_t']:>9.3f} {p['tau_t']:>7.3f} {p['w_cost']:>7.3f}")
