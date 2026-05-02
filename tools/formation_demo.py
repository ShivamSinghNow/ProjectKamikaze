"""KAM-8 host-side spike: 5 drones hovering in cardinal-sentry formation
in a PyBullet GUI window. Visual sanity check that the same SwarmAviary
the sim service uses also renders correctly with gui=True.

Usage (from repo root):
    uv run --extra envs python tools/formation_demo.py
or simply:
    make formation

Requires the kamikaze-common[envs] extra installed in the active venv
(brings pybullet + gym-pybullet-drones).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from kamikaze_common.envs.swarm import (
    DRONE_IDS,
    INITIAL_XYZS,
    NUM_DRONES,
    ROLES,
    SwarmAviary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KAM-8 formation spike")
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="seconds to hover before exit (default 30)",
    )
    p.add_argument(
        "--ctrl_freq",
        type=int,
        default=240,
        help="env control frequency in Hz (default 240)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("[formation_demo] opening PyBullet GUI...")
    print(f"[formation_demo] num_drones={NUM_DRONES} | drone_ids={DRONE_IDS}")
    for drone_id, xyz in zip(DRONE_IDS, INITIAL_XYZS):
        print(f"  {drone_id} ({ROLES[drone_id]:>9}) @ {np.round(xyz, 2).tolist()}")

    env = SwarmAviary(gui=True, ctrl_freq=args.ctrl_freq, pyb_freq=args.ctrl_freq)
    env.reset(seed=0)

    hover_rpms = np.tile(env.HOVER_RPM, (NUM_DRONES, 1)).astype(np.float64)

    n_steps = int(args.duration * args.ctrl_freq)
    print(f"[formation_demo] hovering for {args.duration:.0f}s ({n_steps} steps)...")

    t_start = time.time()
    last_log = t_start
    for step in range(n_steps):
        env.step(hover_rpms)
        # Sleep to keep wall-clock real-time-ish (PyBullet GUI defaults to
        # as-fast-as-possible). 1/ctrl_freq matches sim service.
        time.sleep(1.0 / args.ctrl_freq)
        now = time.time()
        if now - last_log > 5.0:
            elapsed = now - t_start
            print(
                f"[formation_demo] t={elapsed:5.1f}s | "
                f"drone-1 pos={np.round(env.pos[0], 2).tolist()} | "
                f"drone-5 pos={np.round(env.pos[4], 2).tolist()}"
            )
            last_log = now

    print("[formation_demo] done.")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
