#!/usr/bin/env bash
# Launch 3 PPO variants in parallel, one per GPU, on the 3x RTX 3090 box.
# KAM-9 will tune the reward functions; KAM-5 just ships the dispatcher.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

run() {
  local gpu="$1"
  local reward="$2"
  local tag="$3"
  echo ">>> launching reward=$reward on cuda:$gpu (tag=$tag)"
  uv run python -m kamikaze_train.intercept \
    --device "cuda:$gpu" \
    --n_envs 32 \
    --reward "$reward" \
    --run_tag "$tag" \
    > "logs/${tag}.log" 2>&1 &
}

run 0 baseline           baseline
run 1 closing_velocity   closing_velocity
run 2 collision_penalty  collision_penalty

wait
echo ">>> sweep complete. checkpoints in ../models/intercept_ppo_*.zip"
