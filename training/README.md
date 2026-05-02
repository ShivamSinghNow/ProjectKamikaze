# training/

PPO trainer for the `InterceptAviary` environment.

## Single run

```bash
uv sync
uv run python -m kamikaze_train.intercept \
    --device cuda:0 \
    --n_envs 32 \
    --run_tag baseline
```

Writes checkpoint to `../models/intercept_ppo_<run_tag>.zip`.

## 3-GPU sweep (the 3090 box)

```bash
bash scripts/sweep_3gpu.sh
```

Launches three reward variants in parallel on `cuda:0`, `cuda:1`, `cuda:2`:

- `baseline`           — `-distance_to_target + hit_bonus`
- `closing_velocity`   — baseline + closing-velocity shaping
- `collision_penalty`  — baseline + collision/fuel penalty

Eval all three on a held-out scenario set, copy the winning `.zip` to `models/intercept_ppo.zip`, then drone-agent containers will load it automatically when `POLICY=ppo` is set.

## TODO (KAM-9)

- Implement reward variants in `kamikaze_common.envs.intercept.InterceptAviary`
- Implement `train()` in `kamikaze_train/intercept.py`
- Add eval callback that writes scenario sweep stats to `logs/<run_tag>.eval.json`
