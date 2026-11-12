#!/usr/bin/env python
"""Baseline latency and power benchmark for a LeRobot VLA policy.

Measures the unmodified model, run the way LeRobot runs it, so every later
Grail optimization has a fixed number to beat:

  * latency of one inference call (one action chunk), as p50 / p90 / p99 / max
  * the highest control rate the model can sustain synchronously, k / t
  * mean power during the timed loop, and energy per call
  * optionally, time per model stage (vision + embed, backbone prefill, action expert)

It feeds synthetic observations of the right shape, so it measures speed and
power only. Task success is measured separately in LIBERO; see README.md.

Example (Jetson AGX Orin, max power mode):

    sudo nvpmodel -m 0 && sudo jetson_clocks
    python bench_lerobot.py --model lerobot/pi0_libero_finetuned_v044 --profile
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TASK = "put the black bowl on the plate"


# ---------------------------------------------------------------------------
# Power sampling
# ---------------------------------------------------------------------------

# tegrastats prints rails as  NAME current/averagemW  (newer JetPack) or
# NAME current/average  (older). VDD_IN is total board input on AGX Orin.
_RAIL = re.compile(r"(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW|(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+)\s+(\d+)/(\d+)")


def parse_tegrastats(line: str) -> dict[str, float]:
    """Return {rail: watts} for the instantaneous readings in one tegrastats line."""
    rails = {}
    for m in _RAIL.finditer(line):
        name = m.group(1) or m.group(4)
        mw = m.group(2) or m.group(5)
        rails[name] = int(mw) / 1000.0
    return rails


def total_board_watts(rails: dict[str, float]) -> float | None:
    for key in ("VDD_IN", "VIN_SYS_5V0", "POM_5V_IN"):
        if key in rails:
            return rails[key]
    return None


class PowerSampler:
    """Samples power in a background thread while the timed loop runs.

    Uses tegrastats on Jetson (whole-board input power) and nvidia-smi on
    discrete GPUs (GPU power only). Reports None if neither is available.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.source = None
        self.samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._proc = None
        self._thread = None
        if shutil.which("tegrastats"):
            self.source = "tegrastats (board input, VDD_IN)"
        elif shutil.which("nvidia-smi"):
            self.source = "nvidia-smi (GPU only)"

    def _read(self):
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self.source and self.source.startswith("tegrastats"):
                rails = parse_tegrastats(line)
                total = total_board_watts(rails)
                if total is not None:
                    self.samples.append(total)
                gpu = rails.get("VDD_GPU_SOC")
                if gpu is not None:
                    self.gpu_samples.append(gpu)
            else:
                try:
                    self.samples.append(float(line.strip()))
                except ValueError:
                    pass

    def __enter__(self):
        if self.source is None:
            return self
        if self.source.startswith("tegrastats"):
            cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        else:
            cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                   "-lms", str(self.interval_ms)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # let the first samples arrive
        self.samples.clear()
        self.gpu_samples.clear()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None

    def gpu_mean(self) -> float | None:
        return statistics.fmean(self.gpu_samples) if self.gpu_samples else None


# ---------------------------------------------------------------------------
# Stats and environment
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    s = sorted(values)
    if not s:
        raise ValueError("no values")
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_ms(values_s: list[float]) -> dict[str, float]:
    ms = [v * 1000.0 for v in values_s]
    return {
        "mean": statistics.fmean(ms),
        "p50": percentile(ms, 50),
        "p90": percentile(ms, 90),
        "p99": percentile(ms, 99),
        "max": max(ms),
        "std": statistics.pstdev(ms),
    }


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(torch) -> dict:
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import lerobot
        env["lerobot"] = getattr(lerobot, "__version__", None)
    except ImportError:
        pass
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        env["jetson_model"] = model_file.read_text(errors="ignore").strip("\x00\n ")
        env["nvpmodel"] = _run(["nvpmodel", "-q"])
    release = Path("/etc/nv_tegra_release")
    if release.exists():
        env["l4t"] = release.read_text(errors="ignore").splitlines()[0]
    return env


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def synthetic_observation(config, task: str, seed: int):
    """One raw observation (no batch dim) shaped like the policy's inputs."""
    import torch
    from lerobot.configs.types import FeatureType

    g = torch.Generator().manual_seed(seed)
    obs = {}
    for key, feat in config.input_features.items():
        if feat.type == FeatureType.VISUAL:
            obs[key] = torch.rand(*feat.shape, generator=g)  # [C, H, W] in [0, 1]
        elif feat.type == FeatureType.STATE:
            obs[key] = torch.randn(*feat.shape, generator=g)
    obs["task"] = task
    return obs


# Methods on the inner model (policy.model) that bound each stage of a pi0-style
# VLA. sample_actions = embed_prefix + backbone prefill + num_steps * denoise_step.
# LeRobot calls the language model's .forward() directly and walks its layers by
# hand, which bypasses nn.Module hooks, so we time these methods instead.
STAGE_METHODS = {
    "vision_and_embed": "embed_prefix",
    "action_expert": "denoise_step",
}
TOTAL_METHOD = "sample_actions"


class StageTimer:
    """Times model stages with CUDA events by wrapping methods on the instance.

    Each stage's time is summed per inference call, so the action expert
    includes every denoising step. The language backbone prefill is reported as
    the remainder of sample_actions after the other stages.
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

    def _wrap(self, name):
        torch = self.torch
        original = getattr(self.inner, name)
        self.originals[name] = original
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

    def reset(self):
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

    def remove(self):
        for name in self.originals:
            delattr(self.inner, name)  # drop the instance attribute; the class method shows through again


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lerobot/pi0_libero_finetuned_v044")
    ap.add_argument("--policy", default="pi0", help="LeRobot policy type: pi0, pi05, smolvla")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--control-hz", type=float, default=None,
                    help="check the latency budget k/f at this control rate")
    ap.add_argument("--profile", action="store_true", help="also time vision / backbone / action expert")
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = ap.parse_args()

    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu to run (slowly) on CPU")
    cuda = args.device == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    print(f"loading {args.model} ...", flush=True)
    policy_cls = get_policy_class(args.policy)
    policy = policy_cls.from_pretrained(args.model).to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    cfg = policy.config
    k = cfg.n_action_steps
    print(f"chunk_size={cfg.chunk_size} n_action_steps={k}", flush=True)

    obs = synthetic_observation(cfg, args.task, seed=0)

    def call() -> tuple[float, float]:
        """One inference call. Returns (end-to-end seconds, model-only seconds)."""
        sync()
        t0 = time.perf_counter()
        batch = pre(obs)
        sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        sync()
        t2 = time.perf_counter()
        post(actions)
        sync()
        t3 = time.perf_counter()
        return t3 - t0, t2 - t1

    print(f"warmup {args.warmup} ...", flush=True)
    for _ in range(args.warmup):
        call()

    stage_timer = None
    if args.profile and cuda:
        inner = getattr(policy, "model", None)
        stage_timer = StageTimer(inner, torch) if inner is not None else None
        matched = stage_timer.matched if stage_timer else []
        if len(matched) < len(STAGE_METHODS) + 1:
            print(f"--profile: expected pi0-style methods, found {matched or 'none'}; stage split skipped", flush=True)
            if stage_timer is not None:
                stage_timer.remove()
            stage_timer = None

    e2e, model_only, stages = [], [], []
    print(f"timing {args.iters} calls ...", flush=True)
    with PowerSampler() as power:
        loop_start = time.perf_counter()
        for _ in range(args.iters):
            if stage_timer is not None:
                stage_timer.reset()
            a, b = call()
            e2e.append(a)
            model_only.append(b)
            if stage_timer is not None:
                stages.append(stage_timer.collect_ms())
        loop_s = time.perf_counter() - loop_start
    if stage_timer is not None:
        stage_timer.remove()

    lat = summarize_ms(e2e)
    mdl = summarize_ms(model_only)
    watts = power.mean()
    result = {
        "model": args.model,
        "policy": args.policy,
        "device": args.device,
        "chunk_size": cfg.chunk_size,
        "n_action_steps": k,
        "iters": args.iters,
        "warmup": args.warmup,
        "latency_ms": lat,
        "model_only_latency_ms": mdl,
        # Synchronous execution: the robot can run at most k actions per call duration.
        "max_control_hz_p50": k / (lat["p50"] / 1000.0),
        "max_control_hz_p99": k / (lat["p99"] / 1000.0),
        "power_source": power.source,
        "mean_power_w": watts,
        "mean_gpu_power_w": power.gpu_mean(),
        "energy_per_call_j": (watts * loop_s / args.iters) if watts is not None else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if cuda else None,
        "environment": environment(torch),
    }
    if stages:
        keys = [*STAGE_METHODS, "language_backbone_prefill"]
        result["stage_ms_mean"] = {key: statistics.fmean(d[key] for d in stages) for key in keys if all(key in d for d in stages)}
        result["denoise_steps_per_call"] = stages[0].get("denoise_steps")
    if args.control_hz:
        budget_ms = k / args.control_hz * 1000.0
        result["budget"] = {
            "control_hz": args.control_hz,
            "t_max_ms": budget_ms,
            "p99_within_budget": lat["p99"] <= budget_ms,
            "calls_over_budget": sum(1 for v in e2e if v * 1000.0 > budget_ms),
        }

    print()
    print(f"  latency per call   p50 {lat['p50']:8.1f} ms   p99 {lat['p99']:8.1f} ms   max {lat['max']:8.1f} ms")
    print(f"  model only         p50 {mdl['p50']:8.1f} ms   p99 {mdl['p99']:8.1f} ms")
    print(f"  max control rate   {result['max_control_hz_p99']:8.1f} Hz at p99 (k={k}, synchronous)")
    if watts is not None:
        print(f"  mean power         {watts:8.1f} W   ({power.source})")
        print(f"  energy per call    {result['energy_per_call_j']:8.2f} J")
    else:
        print("  power              not measured (no tegrastats or nvidia-smi)")
    if "stage_ms_mean" in result:
        for s, v in result["stage_ms_mean"].items():
            print(f"  stage {s:<17}{v:8.1f} ms")
    if "budget" in result:
        b = result["budget"]
        verdict = "within" if b["p99_within_budget"] else "OVER"
        print(f"  budget at {b['control_hz']:g} Hz     t_max {b['t_max_ms']:.1f} ms, p99 {verdict}, "
              f"{b['calls_over_budget']} of {args.iters} calls over")

    out = args.out or Path(f"baseline-{args.policy}-{datetime.now():%Y%m%d-%H%M%S}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
