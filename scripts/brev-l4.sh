#!/usr/bin/env bash
# First Grail benchmark on a fresh NVIDIA Brev L4 instance.
#
# On the instance:
#   git clone https://github.com/NiLabs-Org/grail && cd grail
#   export HF_TOKEN=hf_...      # a token whose account has accepted the license
#                               # for google/paligemma-3b-pt-224 (pi0's tokenizer)
#   bash scripts/brev-l4.sh             # speed, power, drift, async (~30-60 min)
#   bash scripts/brev-l4.sh --libero    # also LIBERO task success (several hours)
#
# Everything lands in benchmarks/results/l4-<timestamp>/ plus a .tar.gz of it.
set -euo pipefail

MODEL="${MODEL:-lerobot/pi0_libero_finetuned_v044}"
LIBERO=0
[[ "${1:-}" == "--libero" ]] && LIBERO=1
OUT="benchmarks/results/l4-$(date +%Y%m%d-%H%M%S)"

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with access to google/paligemma-3b-pt-224}"
export HF_TOKEN

step() { printf '\n=== %s ===\n' "$*"; }

step "GPU"
nvidia-smi

step "Python 3.12 environment (LeRobot needs >= 3.12)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
EXTRAS="pi"
[[ $LIBERO == 1 ]] && EXTRAS="pi,libero"
uv pip install "lerobot[$EXTRAS]" torchao -e .
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUT"
run() {  # run <name> <grail args...>: log to file and screen, keep going on failure
  local name="$1"; shift
  step "$name: grail $*"
  if ! grail "$@" --out "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"; then
    echo "!! $name failed; see $OUT/$name.log" | tee -a "$OUT/FAILED"
  fi
}

# 1. Baseline: the checkpoint exactly as published (bf16 + torch.compile per its config).
run baseline bench --model "$MODEL" --control-hz 20
# 2. Where the time goes (stage split needs compile off, so these are slower absolute numbers).
run stages bench --model "$MODEL" --no-compile --profile --iters 50
# 3. Candidates, each against the baseline: speed, power and action drift.
run denoise5 compare --model "$MODEL" --denoise-steps 5 --control-hz 20
run denoise3 compare --model "$MODEL" --denoise-steps 3 --control-hz 20
run int8wo   compare --model "$MODEL" --quantize int8wo --control-hz 20
# 4. Async execution at 50 Hz with the published model: count deadline misses.
run async50 async --model "$MODEL" --control-hz 50 --seconds 30

if [[ $LIBERO == 1 ]]; then
  export MUJOCO_GL=egl
  for steps in 10 5; do
    step "LIBERO task success, $steps denoising steps"
    lerobot-eval \
      --policy.path="$MODEL" \
      --policy.num_inference_steps="$steps" \
      --env.type=libero \
      --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
      --eval.batch_size=1 \
      --eval.n_episodes=10 \
      --env.max_parallel_tasks=1 \
      --output_dir="$OUT/libero-steps$steps" 2>&1 | tee "$OUT/libero-steps$steps.log" \
      || echo "!! LIBERO steps=$steps failed" | tee -a "$OUT/FAILED"
  done
fi

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
step "Done"
echo "Results: $OUT"
echo "Archive: $OUT.tar.gz"
echo "Copy it to your machine with:  scp <instance-name>:$(pwd)/$OUT.tar.gz ."
[[ -f "$OUT/FAILED" ]] && { echo "Some steps failed:"; cat "$OUT/FAILED"; }
exit 0
