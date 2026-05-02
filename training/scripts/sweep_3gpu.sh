#!/usr/bin/env bash
# Launch 3 PPO variants in parallel, one per GPU, on the 3x RTX 3090 box.
# KAM-9 will tune the reward functions; KAM-5 just ships the dispatcher.
#
# Failure model: bare `wait` returns the LAST job's status, so a silent OOM
# on cuda:0 paired with a clean cuda:2 finish would exit 0 and the operator
# would not notice until tomorrow's checkpoint is wrong. We track per-pid
# exit codes AND grep each log for "Traceback" so failures surface loudly.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

declare -a pids=()
declare -a tags=()

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
  pids+=("$!")
  tags+=("$tag")
}

run 0 baseline           baseline
run 1 closing_velocity   closing_velocity
run 2 collision_penalty  collision_penalty

# Collect each run's exit code independently — bare `wait` masks failures.
overall_rc=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  tag="${tags[$i]}"
  set +e
  wait "$pid"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "!!! run '$tag' (pid=$pid) FAILED with exit $rc"
    overall_rc=1
  else
    echo "<<< run '$tag' (pid=$pid) ok"
  fi
  # Belt-and-suspenders: even an exit 0 from a misbehaving sb3 callback can
  # leave a Traceback in the log. Treat that as failure.
  if grep -q "Traceback" "logs/${tag}.log"; then
    echo "!!! run '$tag' wrote a Traceback to logs/${tag}.log — flagging failure"
    overall_rc=1
  fi
done

if [ "$overall_rc" -ne 0 ]; then
  echo ">>> sweep FAILED — see logs/*.log; checkpoints are NOT trustworthy"
  exit "$overall_rc"
fi

echo ">>> sweep complete. checkpoints in ../models/intercept_ppo_*.zip"
