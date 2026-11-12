# Grail

Real-time inference for robot foundation models, on the chips inside robots.

> **Pre-alpha.** The measurement harness and async runner work and are tested;
> the first benchmark has not been published yet.

## Why

Robotics is moving to vision-language-action (VLA) models such as π0, OpenVLA,
SmolVLA and GR00T. They are too slow and too power-hungry for onboard hardware,
so teams hand-optimize for weeks, tether robots to big GPUs, or accept jerky
behavior.

Robot inference is not chatbot serving. It runs at batch size 1, on
fixed-length action chunks, against a clock. A model that emits *k* actions per
call on a robot running at *f* Hz must produce the next chunk within

```
t_max = k / f
```

on every call, not on average. At one action and 30 Hz that is 33 ms.

## What is here

| | |
|---|---|
| `grail bench` | Latency (p50 / p99 / max), sustainable control rate, power, energy per call, peak memory, and a per-stage split for a LeRobot policy. |
| `grail compare` | The checkpoint as published vs. the same checkpoint with Grail options: speedup, energy, and **action drift** (same inputs, same diffusion noise). |
| `grail async` | Runs the model through `ChunkRunner` at a fixed control rate and counts deadline misses. |
| `grail.ChunkRunner` | Executes the current action chunk while the next one is computed, drops actions that went stale during inference, and falls back (hold, slow down, stop) instead of blocking when a deadline is missed. Model-agnostic, pure Python. |

Options that `compare` can apply: `--denoise-steps`, `--quantize int8wo`,
`--dtype`, `--compile` / `--no-compile`, `--compile-mode`.

The baseline is always the checkpoint **as published**, with whatever precision
and compile settings its own config sets. For `lerobot/pi0_libero_finetuned_v044`
that is already bf16 with `torch.compile`, so those are not counted as Grail wins.

## Quick start

```bash
git clone https://github.com/NiLabs-Org/grail && cd grail
pip install -e ".[lerobot]"                 # not on PyPI yet; LeRobot needs Python 3.12
export HF_TOKEN=hf_...                       # pi0 uses the gated PaliGemma tokenizer
grail bench   --control-hz 20
grail compare --denoise-steps 5
grail async   --control-hz 50
```

On a fresh NVIDIA Brev L4 instance, [`scripts/brev-l4.sh`](scripts/brev-l4.sh)
installs everything and runs the full first benchmark.

## First target

| | |
|---|---|
| Model | π0 via LeRobot ([`lerobot/pi0_libero_finetuned_v044`](https://huggingface.co/lerobot/pi0_libero_finetuned_v044)) |
| Development hardware | NVIDIA L4 (24 GB, 72 W) on Brev |
| Next | The same benchmark on Jetson-class robot hardware |
| Task success | LIBERO simulation suites |
| Done when | A clear speedup or energy reduction over the published checkpoint, task success within noise of it, reproducible from this repo |

Action drift is a fast filter, not proof. An optimization counts only when task
success in LIBERO (or on the robot) holds.

## Benchmarks

See [`benchmarks/`](benchmarks/) for the methodology.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover everything that runs without a GPU: the runner, statistics,
power parsing, drift and stage timing. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Built by [NiLabs](https://www.nilabs.dev) · [hello@nilabs.dev](mailto:hello@nilabs.dev) · [@NiLabs_org](https://x.com/NiLabs_org)
