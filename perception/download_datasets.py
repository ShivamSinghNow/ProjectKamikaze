"""Pull every enabled source in datasets.yaml into data/raw/<slug>/.

Handles Roboflow, Kaggle, and Hugging Face. Failures are logged and skipped so
one broken slug doesn't block the rest.

Env vars:
    ROBOFLOW_API_KEY   required for Roboflow sources
    KAGGLE_USERNAME    } either both set, or ~/.config/kaggle/kaggle.json present
    KAGGLE_KEY         }
    HF_TOKEN           optional, only needed for gated HF repos
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def slugify(*parts: str) -> str:
    return "__".join(p.replace("/", "_") for p in parts if p)


def pull_roboflow(src: dict, target: Path) -> None:
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(src["workspace"]).project(src["project"])

    requested = src.get("version", "latest")
    if requested in (None, "latest", "auto"):
        versions = list(project.versions())
        nums: list[int] = []
        for v in versions:
            raw = getattr(v, "version", "") or ""
            tail = str(raw).rsplit("/", 1)[-1]
            if tail.isdigit():
                nums.append(int(tail))
        if not nums:
            raise RuntimeError(f"no versions discovered for {src['workspace']}/{src['project']}")
        version_num = max(nums)
        print(f"  -> latest version v{version_num}")
    else:
        version_num = int(requested)

    version = project.version(version_num)
    version.download(src.get("format", "yolov8"), location=str(target))


def pull_kaggle(src: dict, target: Path) -> None:
    import kagglehub

    path = kagglehub.dataset_download(src["slug"], force_download=False)
    # kagglehub puts files in its cache; symlink/copy into target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    target.symlink_to(path)


def pull_huggingface(src: dict, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=src["repo"],
        repo_type=src.get("repo_type", "dataset"),
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
    )


HANDLERS = {
    "roboflow": pull_roboflow,
    "kaggle": pull_kaggle,
    "huggingface": pull_huggingface,
}


def main() -> int:
    cfg = yaml.safe_load((Path(__file__).parent / "datasets.yaml").read_text())
    RAW.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, str]] = []  # (status, slug, msg)
    for src in cfg["sources"]:
        if src.get("enabled") is False:
            continue
        kind = src["type"]
        if kind == "roboflow":
            slug = slugify(kind, src["workspace"], src["project"], f"v{src.get('version', 1)}")
        elif kind == "kaggle":
            slug = slugify(kind, src["slug"])
        else:
            slug = slugify(kind, src["repo"])
        target = RAW / slug

        if target.exists() and any(target.iterdir()):
            print(f"skip   {slug}  (already present)")
            summary.append(("skip", slug, ""))
            continue

        print(f"pull   {slug}  [{kind}]")
        try:
            HANDLERS[kind](src, target)
            summary.append(("ok", slug, ""))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            traceback.print_exc(limit=1)
            summary.append(("fail", slug, str(exc)))

    print("\n=== summary ===")
    for status, slug, msg in summary:
        marker = {"ok": "✓", "skip": "·", "fail": "✗"}[status]
        print(f"  {marker}  {slug}  {msg}")
    failed = sum(1 for s, *_ in summary if s == "fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
