# Benchmarks

Every number Grail publishes comes from a script in this folder and can be
reproduced from it. Speed and power are always reported next to task success.

## What we measure

| Metric | Definition |
|---|---|
| Latency | Wall time of one inference call (one action chunk), preprocessing to unnormalized actions, GPU synchronized. Reported as p50, p90, p99 and max, never only the mean. |
| Model-only latency | The same call without LeRobot's pre- and post-processing. |
| Max control rate | k / latency, for chunk length k, running synchronously. Reported at p50 and p99. |
| Latency budget | t_max = k / f at a given control rate f, and how many calls exceeded it. |
| Power | Mean board input power during the timed loop (`VDD_IN` from `tegrastats` on Jetson), plus GPU rail power. On discrete GPUs, GPU power from `nvidia-smi`. |
| Energy per call | Mean power × mean time per call. |
| Stage split | Vision encoder + embedding, language backbone prefill, action expert (all denoising steps), from CUDA events. |
| Task success | Success rate on the LIBERO suites, from `lerobot-eval`. |

## Rules

- Same hardware, power mode and clocks for every number being compared.
  On Jetson, record `nvpmodel -q` and whether `jetson_clocks` was on.
- Warm up before timing (20 calls by default) and time at least 200 calls.
- Nothing else running on the device.
- A result is only a win if task success stays within noise of the baseline.

## Baseline: π0 on Jetson AGX Orin

**Setup.** JetPack 6, PyTorch for Jetson, then LeRobot with the π0 and LIBERO extras:

```bash
pip install "lerobot[pi,libero]"
```

**Speed and power** (no simulator needed; uses synthetic inputs of the right shape):

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
python benchmarks/baseline/bench_lerobot.py \
  --model lerobot/pi0_libero_finetuned_v044 \
  --control-hz 20 --profile \
  --out benchmarks/results/pi0-orin-baseline.json
```

**Task success** on the four standard LIBERO suites:

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned_v044 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```

LIBERO renders with MuJoCo. If it is too slow on the Orin itself, run the
success evaluation on a workstation with the same checkpoint and the same
LeRobot version, and keep the Orin for speed and power.

## Results

None yet. Results land in [`results/`](results/) as JSON, with the command that produced them.
