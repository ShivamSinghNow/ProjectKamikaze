# ProjectKamikaze

Counter-Shahed swarm drone hackathon (3rd Annual NatSec Hackathon, May 2-3 2026). Five PyBullet drones run YOLO26 perception, gossip detections over a mesh, and intercept the threat. Demo orchestrated through Palantir AIP.

## Repo layout

```
services/
  sim/                # central PyBullet sim (KAM-8) — 30Hz physics, pub world:state
  drone-agent/        # per-drone container: perception + policy + mesh + sim_client
  control-plane/      # FastAPI roster + heartbeat eviction + (later) AIP bridge
packages/
  kamikaze_common/    # shared schemas, logging, gym envs (SwarmAviary), redis_io
training/
  kamikaze_train/     # PPO trainer for the InterceptAviary (KAM-9)
tools/
  formation_demo.py   # KAM-8 host-side spike (PyBullet GUI)
models/               # ONNX + PPO checkpoints (gitignored)
```

## Formation

```
        drone-1  (N)        Drones 1-4 = perimeter, altitude 1.5m,
            ↑                cameras facing outward (90° each).
drone-4 ←  d-5  → drone-2   Drone 5 = spotter, altitude 5m, 360° pano (KAM-10).
            ↓                Picked for omnidirectional Shahed detection +
        drone-3  (S)         graceful kill-a-drone demo.
```

## Quick start

```bash
# Containerized swarm: redis + control-plane + sim + 5 drones
make up
make logs            # tail everything
make sim-logs        # sim service only
make roster          # curl /roster
make formation       # host-side PyBullet GUI (5 drones in formation, 30s)
make down
```

The default `POLICY=noop` makes drones hover (sim physics is real but the policy returns hover RPMs) — no checkpoint required. KAM-9 will flip the default to `ppo` once `models/intercept_ppo.zip` exists.

## Sim dataplane (KAM-8)

The `sim` service runs `SwarmAviary(num_drones=5)` headless at 30Hz. Per tick:

1. Drains pending `drone:{id}:cmd` Redis messages, updates motor RPMs.
2. Steps physics.
3. Publishes `world:state` (msgpack: poses for all drones).

Each `drone-agent` container subscribes to `world:state`, extracts its own pose, runs perception + policy, publishes its 4-RPM motor command back. KAM-10 will extend the `world:state` payload with per-drone camera frames.

Channel/payload definitions live in `packages/kamikaze_common/redis_io.py` so the sim and the drone-agent can never drift on protocol.

## Tickets

- KAM-5  scaffold (PR #1)
- KAM-6  Shahed dataset prep
- KAM-7  YOLO26-n training + ONNX export
- KAM-8  gym-pybullet-drones spike + central sim service (this PR)
- KAM-9  InterceptAviary + PPO training (3-GPU sweep)
- KAM-10 wire YOLO26 ONNX into drone agent (incl. spotter 360° pano)
- KAM-11 gossip mesh + consensus fusion
- KAM-12 Palantir Foundry/AIP HUD
- KAM-13 end-to-end demo + 90s pitch

System diagram: https://excalidraw.com/#json=Smxs7HGKruLH0St-3Z1Jg,_2yvfJHuOfSQKG26lJntyw
