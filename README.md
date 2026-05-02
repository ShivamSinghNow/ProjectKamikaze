# ProjectKamikaze

Counter-Shahed swarm drone hackathon (3rd Annual NatSec Hackathon, May 2-3 2026). Five PyBullet drones run YOLO26 perception, gossip detections over a mesh, and intercept the threat. Demo orchestrated through Palantir AIP.

## Repo layout

```
services/
  drone-agent/        # per-drone container: perception + policy + mesh + sim
  control-plane/      # FastAPI roster + AIP bridge
packages/
  kamikaze_common/    # shared schemas, logging, gym envs
training/
  kamikaze_train/     # PPO trainer for the InterceptAviary
models/               # ONNX + PPO checkpoints (gitignored)
```

## Quick start

```bash
# Bring up redis + control-plane + 5 drones
make up

# Tail logs
make logs

# Tear down
make down
```

The default `POLICY=noop` makes drones hover and emit heartbeats — no checkpoint required. KAM-9 will flip the default to `ppo` once `models/intercept_ppo.zip` exists.

## Tickets

- KAM-5  scaffold (this PR)
- KAM-6  Shahed dataset prep
- KAM-7  YOLO26-n training + ONNX export
- KAM-8  gym-pybullet-drones spike (5 drones in formation)
- KAM-9  InterceptAviary + PPO training (3-GPU sweep)
- KAM-10 wire YOLO26 ONNX into drone agent
- KAM-11 gossip mesh + consensus fusion
- KAM-12 Palantir Foundry/AIP HUD
- KAM-13 end-to-end demo + 90s pitch

System diagram: https://excalidraw.com/#json=Smxs7HGKruLH0St-3Z1Jg,_2yvfJHuOfSQKG26lJntyw
