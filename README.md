# Grail

Real-time inference for robot foundation models, on the chips inside robots.

> **Pre-release.** Nothing here is ready to use yet. The first thing to land is a
> reproducible baseline benchmark; watch the repo to see it.

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

## Goals

- **Built for the robot workload:** batch size 1, fixed shapes, worst-case latency.
- **Optimized per stage:** vision encoder, language backbone and action head.
- **Asynchronous by default:** the robot keeps moving while the next chunk computes.
- **Any chip:** not only NVIDIA.
- **Task success, always:** every speed or power number is reported next to the
  task success of the original model. A faster model that stops completing the
  task does not count.

## First target

| | |
|---|---|
| Model | π0, via LeRobot ([`lerobot/pi0_libero_finetuned_v044`](https://huggingface.co/lerobot/pi0_libero_finetuned_v044)) |
| Hardware | NVIDIA Jetson AGX Orin 64GB |
| Baseline | Stock PyTorch through LeRobot |
| Task success | LIBERO simulation suites |
| Done when | A clear speedup or power reduction, task success within noise of the baseline, reproducible from this repo |

## Benchmarks

[`benchmarks/`](benchmarks/) holds the methodology and scripts. The
[baseline script](benchmarks/baseline/) measures latency percentiles, the
control rate the model can sustain, power and energy per call for the
unmodified model.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results from hardware we do
not have are especially welcome.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Built by [NiLabs](https://www.nilabs.dev) · [hello@nilabs.dev](mailto:hello@nilabs.dev) · [@NiLabs_org](https://x.com/NiLabs_org)
