"""Timing, power and stage split for one loaded model."""

from __future__ import annotations

import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .power import PowerSampler
from .stats import summarize_ms

# Methods on the inner model (policy.model) that bound each stage of a pi0-style
# VLA: sample_actions = embed_prefix + backbone prefill + num_steps * denoise_step.
# LeRobot calls the language model's .forward() directly and walks its layers by
# hand, which bypasses nn.Module hooks, so we time these methods instead.
STAGE_METHODS = {
    "vision_and_embed": "embed_prefix",
    "action_expert": "denoise_step",
}
TOTAL_METHOD = "sample_actions"


class StageTimer:
    """Times model stages with CUDA events by wrapping methods on the instance.

    Each stage is summed per inference call, so the action expert includes every
    denoising step; the backbone prefill is the remainder of sample_actions.
    Does not work under torch.compile, which traces through the wrappers.
    """

    def __init__(self, inner, torch):
        self.torch = torch
        self.inner = inner
        self.events: dict[str, list] = {}
        self.originals: dict[str, object] = {}
        for name in [*STAGE_METHODS.values(), TOTAL_METHOD]:
            if callable(getattr(inner, name, None)):
                self._wrap(name)
        self.matched = sorted(self.originals)

    @property
    def complete(self) -> bool:
        return len(self.matched) == len(STAGE_METHODS) + 1

    def _wrap(self, name: str) -> None:
        torch = self.torch
        original = getattr(self.inner, name)
        had_instance_attr = name in vars(self.inner)
        self.originals[name] = (original, had_instance_attr)
        self.events[name] = []

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(*args, **kwargs)
            end.record()
            self.events[name].append((start, end))
            return out

        setattr(self.inner, name, timed)

    def reset(self) -> None:
        for v in self.events.values():
            v.clear()

    def collect_ms(self) -> dict[str, float]:
        self.torch.cuda.synchronize()
        per = {n: sum(s.elapsed_time(e) for s, e in pairs) for n, pairs in self.events.items() if pairs}
        out = {stage: per[m] for stage, m in STAGE_METHODS.items() if m in per}
        if TOTAL_METHOD in per and len(out) == len(STAGE_METHODS):
            out["language_backbone_prefill"] = per[TOTAL_METHOD] - sum(out.values())
        if "denoise_step" in self.events:
            out["denoise_steps"] = len(self.events["denoise_step"])
        return out

    def remove(self) -> None:
        for name, (original, had_instance_attr) in self.originals.items():
            if had_instance_attr:
                setattr(self.inner, name, original)
            else:
                delattr(self.inner, name)


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(torch=None) -> dict:
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch is not None:
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["device_name"] = torch.cuda.get_device_name(0)
    try:
        import lerobot

        env["lerobot"] = getattr(lerobot, "__version__", None)
    except ImportError:
        pass
    from . import __version__

    env["grail"] = __version__
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                "--format=csv,noheader"])
    if smi:
        env["nvidia_smi"] = smi
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        env["jetson_model"] = model_file.read_text(errors="ignore").strip("\x00\n ")
        env["nvpmodel"] = _run(["nvpmodel", "-q"])
    return env


def measure(model, *, warmup: int = 20, iters: int = 200, profile: bool = False,
            control_hz: float | None = None, log=print) -> dict:
    """Time ``iters`` inference calls of a LeRobotModel after ``warmup`` calls."""
    torch = model.torch
    cuda = model.device.startswith("cuda")
    obs = model.observation(seed=0)
    k = model.config.n_action_steps

    log(f"  warmup {warmup} calls (the first can take minutes with torch.compile) ...")
    t0 = time.perf_counter()
    for _ in range(warmup):
        model.timed_call(obs)
    warmup_s = time.perf_counter() - t0

    stage_timer = None
    if profile and cuda and model.inner is not None:
        if model.effective.get("compile_model"):
            log("  --profile skipped: stage timing does not work under torch.compile")
        else:
            stage_timer = StageTimer(model.inner, torch)
            if not stage_timer.complete:
                log(f"  --profile skipped: expected pi0-style methods, found {stage_timer.matched or 'none'}")
                stage_timer.remove()
                stage_timer = None

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    e2e, model_only, stages = [], [], []
    log(f"  timing {iters} calls ...")
    with PowerSampler() as power:
        loop_start = time.perf_counter()
        for _ in range(iters):
            if stage_timer is not None:
                stage_timer.reset()
            a, b = model.timed_call(obs)
            e2e.append(a)
            model_only.append(b)
            if stage_timer is not None:
                stages.append(stage_timer.collect_ms())
        loop_s = time.perf_counter() - loop_start
    if stage_timer is not None:
        stage_timer.remove()

    lat = summarize_ms(e2e)
    watts = power.mean()
    result = {
        "model": model.model_id,
        "policy": model.policy_type,
        "device": model.device,
        "options": model.options.changed(),
        "effective_config": model.effective,
        "chunk_size": model.config.chunk_size,
        "n_action_steps": k,
        "warmup_calls": warmup,
        "warmup_s": warmup_s,
        "latency_ms": lat,
        "model_only_latency_ms": summarize_ms(model_only),
        "max_control_hz_p50": k / (lat["p50"] / 1000.0),
        "max_control_hz_p99": k / (lat["p99"] / 1000.0),
        "power_source": power.source,
        "mean_power_w": watts,
        "mean_gpu_power_w": power.gpu_mean(),
        "energy_per_call_j": (watts * loop_s / iters) if watts is not None else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if cuda else None,
    }
    if stages:
        keys = [*STAGE_METHODS, "language_backbone_prefill"]
        result["stage_ms_mean"] = {key: statistics.fmean(d[key] for d in stages)
                                   for key in keys if all(key in d for d in stages)}
        result["denoise_steps_per_call"] = stages[0].get("denoise_steps")
    if control_hz:
        budget_ms = k / control_hz * 1000.0
        result["budget"] = {
            "control_hz": control_hz,
            "t_max_ms": budget_ms,
            "p99_within_budget": lat["p99"] <= budget_ms,
            "calls_over_budget": sum(1 for v in e2e if v * 1000.0 > budget_ms),
        }
    return result


def format_result(r: dict) -> str:
    lat, mdl = r["latency_ms"], r["model_only_latency_ms"]
    lines = [
        f"  latency per call   p50 {lat['p50']:8.1f} ms   p99 {lat['p99']:8.1f} ms   max {lat['max']:8.1f} ms",
        f"  model only         p50 {mdl['p50']:8.1f} ms   p99 {mdl['p99']:8.1f} ms",
        f"  max control rate   {r['max_control_hz_p99']:8.1f} Hz at p99 (k={r['n_action_steps']}, synchronous)",
    ]
    if r.get("mean_power_w") is not None:
        lines.append(f"  mean power         {r['mean_power_w']:8.1f} W   ({r['power_source']})")
        lines.append(f"  energy per call    {r['energy_per_call_j']:8.2f} J")
    else:
        lines.append("  power              not measured (no tegrastats or nvidia-smi)")
    if r.get("peak_memory_gb") is not None:
        lines.append(f"  peak memory        {r['peak_memory_gb']:8.2f} GB")
    for s, v in r.get("stage_ms_mean", {}).items():
        lines.append(f"  stage {s:<17}{v:8.1f} ms")
    if "budget" in r:
        b = r["budget"]
        verdict = "within" if b["p99_within_budget"] else "OVER"
        lines.append(f"  budget at {b['control_hz']:g} Hz     t_max {b['t_max_ms']:.1f} ms, p99 {verdict}, "
                     f"{b['calls_over_budget']} calls over")
    return "\n".join(lines)
