"""Grail: real-time inference for robot foundation models."""

from .runner import ChunkRunner, RunnerStats
from .stats import percentile, summarize_ms

__version__ = "0.1.0.dev0"

__all__ = ["ChunkRunner", "RunnerStats", "percentile", "summarize_ms", "__version__"]
