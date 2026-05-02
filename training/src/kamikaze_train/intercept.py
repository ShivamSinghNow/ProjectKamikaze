"""PPO trainer for InterceptAviary.

KAM-5 ships argparse + import scaffolding so the 3-GPU sweep script can be
validated. KAM-9 fills in `train()` with the real PPO config + eval callback.

Usage:
    python -m kamikaze_train.intercept --device cuda:0 --n_envs 32 --run_tag baseline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on InterceptAviary")
    p.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    p.add_argument("--n_envs", type=int, default=32, help="parallel envs in SubprocVecEnv")
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--run_tag", default="baseline", help="filename suffix for ckpt + tb log")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument(
        "--reward",
        choices=["baseline", "closing_velocity", "collision_penalty"],
        default="baseline",
        help="Reward shape — one per GPU in the 3-way sweep (KAM-9 fills these in).",
    )
    return p.parse_args()


def train(args: argparse.Namespace) -> Path:
    """KAM-9: build SubprocVecEnv(InterceptAviary), wrap PPO, learn, save."""
    from kamikaze_common.envs.intercept import InterceptAviary  # noqa: F401 — imports validate

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = Path(args.model_dir) / f"intercept_ppo_{args.run_tag}.zip"

    raise NotImplementedError(
        f"KAM-9: implement train() — would write to {out_path} "
        f"on device={args.device}, n_envs={args.n_envs}, reward={args.reward}"
    )


def main() -> int:
    args = parse_args()
    print(
        f"[kamikaze_train.intercept] device={args.device} "
        f"n_envs={args.n_envs} run_tag={args.run_tag} reward={args.reward} "
        f"timesteps={args.total_timesteps}"
    )
    out = train(args)
    print(f"[kamikaze_train.intercept] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
