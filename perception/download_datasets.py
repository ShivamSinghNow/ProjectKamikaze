"""Download all Shahed datasets listed in datasets.yaml into data/raw/."""
import os
import sys
from pathlib import Path

import yaml
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set. Get one at https://app.roboflow.com/settings/api")

    cfg = yaml.safe_load((Path(__file__).parent / "datasets.yaml").read_text())
    rf = Roboflow(api_key=api_key)
    RAW.mkdir(parents=True, exist_ok=True)

    for src in cfg["sources"]:
        slug = f"{src['workspace']}__{src['project']}"
        target = RAW / slug
        if target.exists():
            print(f"skip  {slug}  (already downloaded)")
            continue
        print(f"pull  {slug}")
        try:
            project = rf.workspace(src["workspace"]).project(src["project"])
            project.version(src["version"]).download(cfg["output_format"], location=str(target))
        except Exception as exc:
            print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
