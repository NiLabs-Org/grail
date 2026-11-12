"""Action drift: how far an optimized model's actions move from the original's.

Both models are run on the same observations with the same diffusion noise, so
any difference comes from the optimization. Drift is a fast proxy, not the
final word: an optimization only counts if task success in simulation (or on
the robot) stays within noise of the original. Drift tells you which candidates
are worth that slower evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# Provisional. Calibrate against task success once both are measured on the
# same candidates; until then treat the verdict as a hint.
DEFAULT_MAX_REL_L2 = 0.05


def _flatten(x) -> list[float]:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().float().cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    out: list[float] = []
    stack = [x]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(float(item))
    return out


def action_drift(reference, candidate) -> dict[str, float]:
    """Compare two action chunks of the same shape."""
    r, c = _flatten(reference), _flatten(candidate)
    if len(r) != len(c):
        raise ValueError(f"shape mismatch: {len(r)} vs {len(c)} values")
    if not r:
        raise ValueError("empty action chunks")
    diff = [a - b for a, b in zip(c, r)]
    r_norm = math.sqrt(sum(v * v for v in r))
    c_norm = math.sqrt(sum(v * v for v in c))
    d_norm = math.sqrt(sum(v * v for v in diff))
    dot = sum(a * b for a, b in zip(r, c))
    return {
        "max_abs": max(abs(v) for v in diff),
        "mean_abs": sum(abs(v) for v in diff) / len(diff),
        "rel_l2": d_norm / r_norm if r_norm > 0 else (0.0 if d_norm == 0 else math.inf),
        "cosine": dot / (r_norm * c_norm) if r_norm > 0 and c_norm > 0 else (1.0 if d_norm == 0 else 0.0),
    }


def summarize_drift(
    per_sample: Iterable[dict[str, float]], max_rel_l2: float = DEFAULT_MAX_REL_L2
) -> dict:
    """Worst case across samples, plus a provisional pass/fail on rel_l2."""
    samples: Sequence[dict[str, float]] = list(per_sample)
    if not samples:
        raise ValueError("no samples")
    worst = {
        "max_abs": max(s["max_abs"] for s in samples),
        "mean_abs": max(s["mean_abs"] for s in samples),
        "rel_l2": max(s["rel_l2"] for s in samples),
        "cosine": min(s["cosine"] for s in samples),
    }
    return {
        "samples": len(samples),
        "worst": worst,
        "max_rel_l2": max_rel_l2,
        "within_tolerance": worst["rel_l2"] <= max_rel_l2,
        "note": "proxy only; confirm with task success",
    }
