# Benchmarks

Every number Grail publishes comes from the `grail` commands and can be
reproduced from them. Speed and power are always reported next to task success.

## What we measure

| Metric | Definition |
|---|---|
| Latency | Wall time of one inference call (one action chunk), preprocessing to unnormalized actions, GPU synchronized. Reported as p50, p90, p99 and max, never only the mean. |
| Model-only latency | The same call without LeRobot's pre- and post-processing. |
| Max control rate | k / latency, for chunk length k, running synchronously. Reported at p50 and p99. |
| Latency budget | t_max = k / f at a given control rate f, and how many calls exceeded it. |
| Power | Mean power during the timed loop: board input (`VDD_IN` via `tegrastats`) on Jetson, GPU power via `nvidia-smi` elsewhere. GPU-only power understates whole-system power. |
| Energy per call | Mean power × mean time per call. |
| Stage split | Vision encoder + embedding, language backbone prefill, action expert (all denoising steps), from CUDA events. Needs `--no-compile`, so its absolute numbers are slower than the baseline. |
| Action drift | Same observations and same diffusion noise through the baseline and the candidate; worst-case relative L2, max absolute error and cosine similarity over the samples. A filter for candidates, not a substitute for task success. |
| Async misses | Control steps with no fresh action when the model runs through `ChunkRunner` at a given rate. |
| Task success | Success rate on the LIBERO suites, from `lerobot-eval`. |

## Rules

- The baseline is the checkpoint as published, with its own dtype and compile settings.
- Same hardware, driver, power limit and clocks for every number being compared.
  Results record `nvidia-smi` (or `nvpmodel -q` on Jetson) automatically.
- Warm up before timing (20 calls by default; `torch.compile` makes the first slow) and time at least 200 calls.
- Nothing else running on the device.
- A result is only a win if task success stays within noise of the baseline.

## π0 on an NVIDIA L4 (Brev)

```bash
git clone https://github.com/NiLabs-Org/grail && cd grail
export HF_TOKEN=hf_...          # account must have accepted google/paligemma-3b-pt-224
bash scripts/brev-l4.sh         # bench, stage split, three candidates, async at 50 Hz
bash scripts/brev-l4.sh --libero  # also LIBERO task success at 10 and 5 denoising steps
```

The script writes JSON and logs to `benchmarks/results/l4-<timestamp>/` and a
`.tar.gz` of the folder.

Individual commands:

```bash
grail bench   --control-hz 20                          # the published checkpoint
grail bench   --no-compile --profile --iters 50        # where the time goes
grail compare --denoise-steps 5 --control-hz 20        # candidate vs baseline
grail compare --quantize int8wo                        # needs torchao
grail async   --control-hz 50 --seconds 30             # misses under async execution
```

Task success on the four standard LIBERO suites:

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned_v044 \
  --policy.num_inference_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```

## Results

None yet. Results land in [`results/`](results/) as JSON, with the command that produced them.
