# SwarmSight

Counter-Shahed swarm drone hackathon project (3rd Annual NatSec Hackathon, May 2026).

- Linear: https://linear.app/kamikaze/project/swarmsight-cafa3a1b7e83
- Diagram: https://excalidraw.com/#json=Smxs7HGKruLH0St-3Z1Jg,_2yvfJHuOfSQKG26lJntyw

## Layout

- `perception/` — YOLO26 fine-tuning on Shahed datasets (Track A)
- `sim/` — gym-pybullet-drones runtime + intercept env (Track B)
- `mesh/` — gossip + consensus between drone agents (Track B)
- `ops/` — Palantir AIP integration (Track B)
- `data/raw/`, `data/merged/` — gitignored dataset working dirs

## Track A — Perception (KAM-6, KAM-7, KAM-10, KAM-13)

```bash
cd perception
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ROBOFLOW_API_KEY=...   # https://app.roboflow.com/settings/api
python download_datasets.py   # pulls into ../data/raw/
python merge_datasets.py      # produces ../data/merged/data.yaml
```
