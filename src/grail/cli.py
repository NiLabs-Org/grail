"""grail command line.

  grail bench    time one model (the checkpoint as published, or with options)
  grail compare  baseline vs Grail options: speed, power and action drift
  grail async    drive the async chunk runner with the real model and count misses
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "lerobot/pi0_libero_finetuned_v044"


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local path")
    p.add_argument("--policy", default="pi0", help="LeRobot policy type: pi0, pi05, smolvla")
    p.add_argument("--device", default="cuda")


def _add_option_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Grail options (default: leave the checkpoint's setting)")
    g.add_argument("--dtype", choices=["bfloat16", "float32"])
    g.add_argument("--compile", dest="compile", action="store_true", default=None)
    g.add_argument("--no-compile", dest="compile", action="store_false")
    g.add_argument("--compile-mode")
    g.add_argument("--denoise-steps", type=int, help="flow-matching denoising steps (pi0 default 10)")
    g.add_argument("--quantize", choices=["int8wo"], help="int8 weight-only quantization (needs torchao)")


def _options(args):
    from .lerobot_model import Options

    return Options(dtype=args.dtype, compile=args.compile, compile_mode=args.compile_mode,
                   denoise_steps=args.denoise_steps, quantize=args.quantize)


def _write(result: dict, out: Path | None, prefix: str) -> None:
    out = out or Path(f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


def cmd_bench(args) -> int:
    from .bench import environment, format_result, measure
    from .lerobot_model import LeRobotModel

    opts = _options(args)
    print(f"loading {args.model} ({opts.changed() or 'as published'}) ...", flush=True)
    model = LeRobotModel(args.model, args.policy, args.device, opts)
    print(f"  effective: {model.effective}")
    r = measure(model, warmup=args.warmup, iters=args.iters, profile=args.profile, control_hz=args.control_hz)
    r["environment"] = environment(model.torch)
    print()
    print(format_result(r))
    _write(r, args.out, f"bench-{args.policy}")
    return 0


def cmd_compare(args) -> int:
    from .bench import environment, format_result, measure
    from .lerobot_model import LeRobotModel, Options
    from .verify import action_drift, summarize_drift

    opts = _options(args)
    if not opts.changed():
        print("compare needs at least one Grail option, e.g. --denoise-steps 5", file=sys.stderr)
        return 2
    seeds = range(args.samples)

    print(f"[1/2] baseline: {args.model} as published", flush=True)
    base = LeRobotModel(args.model, args.policy, args.device, Options())
    print(f"  effective: {base.effective}")
    ref_actions = [base.predict(base.observation(s), base.noise(s)).cpu() for s in seeds]
    base_r = measure(base, warmup=args.warmup, iters=args.iters, control_hz=args.control_hz)
    env = environment(base.torch)
    base.close()

    print(f"\n[2/2] grail: {opts.changed()}", flush=True)
    cand = LeRobotModel(args.model, args.policy, args.device, opts)
    print(f"  effective: {cand.effective}")
    cand_actions = [cand.predict(cand.observation(s), cand.noise(s)).cpu() for s in seeds]
    cand_r = measure(cand, warmup=args.warmup, iters=args.iters, control_hz=args.control_hz)
    cand.close()

    drift = summarize_drift([action_drift(r, c) for r, c in zip(ref_actions, cand_actions)],
                            max_rel_l2=args.max_rel_l2)
    b, c = base_r["latency_ms"], cand_r["latency_ms"]
    speedup = {"p50": b["p50"] / c["p50"], "p99": b["p99"] / c["p99"]}
    energy = None
    if base_r.get("energy_per_call_j") and cand_r.get("energy_per_call_j"):
        energy = 1 - cand_r["energy_per_call_j"] / base_r["energy_per_call_j"]

    print("\nbaseline\n" + format_result(base_r))
    print("\ngrail\n" + format_result(cand_r))
    print(f"\n  speedup            {speedup['p50']:.2f}x at p50, {speedup['p99']:.2f}x at p99")
    if energy is not None:
        word = "lower" if energy >= 0 else "higher"
        print(f"  energy per call    {abs(energy) * 100:.1f}% {word} than baseline")
    w = drift["worst"]
    verdict = "within" if drift["within_tolerance"] else "OUTSIDE"
    print(f"  action drift       worst rel_l2 {w['rel_l2']:.4f}, max_abs {w['max_abs']:.4f}, "
          f"cosine {w['cosine']:.5f} over {drift['samples']} samples: {verdict} {drift['max_rel_l2']}")
    print("  next: confirm task success in LIBERO before calling this a win")

    _write({"baseline": base_r, "grail": cand_r, "speedup": speedup, "energy_saving": energy,
            "drift": drift, "environment": env}, args.out, f"compare-{args.policy}")
    return 0


def cmd_async(args) -> int:
    from .lerobot_model import LeRobotModel
    from .runner import ChunkRunner

    opts = _options(args)
    print(f"loading {args.model} ({opts.changed() or 'as published'}) ...", flush=True)
    model = LeRobotModel(args.model, args.policy, args.device, opts)
    obs = model.observation(0)
    print(f"  warmup {args.warmup} calls ...", flush=True)
    for _ in range(args.warmup):
        model.timed_call(obs)

    def predict(o):
        return model.predict(o)[0].cpu().tolist()  # list of k action vectors

    sent = []
    with ChunkRunner(predict, control_hz=args.control_hz, refill_at=args.refill_at) as runner:
        print(f"  running {args.seconds:g} s at {args.control_hz:g} Hz ...", flush=True)
        stats = runner.run(lambda: obs, sent.append, duration_s=args.seconds)
    d = stats.as_dict()
    print(f"\n  steps {d['steps']}, misses {d['misses']} ({d['miss_rate'] * 100:.2f}%), "
          f"chunks {d['chunks']}, stale actions dropped {d['stale_actions_dropped']}, "
          f"tick overruns {d['tick_overruns']}")
    if "inference_ms" in d:
        im = d["inference_ms"]
        print(f"  inference while running: p50 {im['p50']:.1f} ms, p99 {im['p99']:.1f} ms, max {im['max']:.1f} ms")
    d.update(model=args.model, options=opts.changed(), control_hz=args.control_hz, seconds=args.seconds)
    _write(d, args.out, f"async-{args.policy}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="grail", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="time one model")
    _add_model_args(b)
    _add_option_args(b)
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--iters", type=int, default=200)
    b.add_argument("--control-hz", type=float, help="check the latency budget k/f at this rate")
    b.add_argument("--profile", action="store_true", help="per-stage split (needs --no-compile)")
    b.add_argument("--out", type=Path)
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("compare", help="baseline vs Grail options")
    _add_model_args(c)
    _add_option_args(c)
    c.add_argument("--samples", type=int, default=8, help="observations for the drift check")
    c.add_argument("--max-rel-l2", type=float, default=0.05)
    c.add_argument("--warmup", type=int, default=20)
    c.add_argument("--iters", type=int, default=200)
    c.add_argument("--control-hz", type=float)
    c.add_argument("--out", type=Path)
    c.set_defaults(func=cmd_compare)

    a = sub.add_parser("async", help="async chunk runner with the real model")
    _add_model_args(a)
    _add_option_args(a)
    a.add_argument("--control-hz", type=float, default=50)
    a.add_argument("--seconds", type=float, default=20)
    a.add_argument("--refill-at", type=int, help="actions left when the next inference starts (default: auto)")
    a.add_argument("--warmup", type=int, default=10)
    a.add_argument("--out", type=Path)
    a.set_defaults(func=cmd_async)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
