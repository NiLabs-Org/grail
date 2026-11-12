"""Latency statistics. Worst cases matter more than averages, so every summary
reports tail percentiles and the maximum."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_ms(values_s: Sequence[float]) -> dict[str, float]:
    """Summarize durations given in seconds, reported in milliseconds."""
    ms = [v * 1000.0 for v in values_s]
    return {
        "mean": statistics.fmean(ms),
        "p50": percentile(ms, 50),
        "p90": percentile(ms, 90),
        "p99": percentile(ms, 99),
        "max": max(ms),
        "std": statistics.pstdev(ms),
        "n": len(ms),
    }
