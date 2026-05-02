"""Merge data/raw/* YOLO datasets into data/merged/ as single-class `shahed`.

Walks each raw dataset, parses its `data.yaml` (or `_classes.txt`) to map source
class indexes to names, drops anything not in `class_aliases`, hashes images to
drop dupes, and emits a YOLO `data.yaml` with one class.

Modality (rgb|ir) is preserved as a directory so we can train on the union or
filter to thermal-only for night-ops eval.
"""
from __future__ import annotations

import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "merged"
CFG = yaml.safe_load((Path(__file__).parent / "datasets.yaml").read_text())

CLASS_ID = 0
CLASS_NAME = CFG["target_class"]
ALIASES = {a.lower() for a in CFG["class_aliases"]}
SPLIT_RATIO = 0.85  # train/val


def alias(name: str) -> bool:
    return name.lower().strip() in ALIASES


def parse_yolo_yaml(p: Path) -> dict[int, str]:
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text())
    names = data.get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {}


def parse_classes_txt(p: Path) -> dict[int, str]:
    if not p.exists():
        return {}
    return {i: line.strip() for i, line in enumerate(p.read_text().splitlines()) if line.strip()}


def find_class_map(ds: Path) -> dict[int, str]:
    for cand in [ds / "data.yaml", ds / "data.yml", ds / "dataset.yaml"]:
        m = parse_yolo_yaml(cand)
        if m:
            return m
    for cand in [ds / "train" / "_classes.txt", ds / "_classes.txt"]:
        m = parse_classes_txt(cand)
        if m:
            return m
    return {}


def relabel(label_text: str, src_map: dict[int, str]) -> str:
    out = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        name = src_map.get(cid, "")
        if not alias(name):
            continue  # drop non-shahed boxes
        out.append(" ".join([str(CLASS_ID), *parts[1:5]]))
    return "\n".join(out)


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def gather_modality(slug: str) -> str:
    for src in CFG["sources"]:
        kind = src["type"]
        if kind == "roboflow":
            s = "roboflow__" + src["workspace"] + "__" + src["project"]
        elif kind == "kaggle":
            s = "kaggle__" + src["slug"].replace("/", "_")
        else:
            s = "huggingface__" + src["repo"].replace("/", "_")
        if s in slug:
            return src.get("modality", "rgb")
    return "rgb"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} — run download_datasets.py first")

    seen: set[str] = set()
    rows: list[tuple[Path, str, str, str]] = []  # (img, label_text, slug, modality)

    for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
        src_map = find_class_map(ds)
        if not src_map:
            print(f"!  {ds.name}: no class map; assume single class 0=shahed")
            src_map = {0: CLASS_NAME}

        modality = gather_modality(ds.name)
        per_ds = 0
        for split in ("train", "valid", "test"):
            img_dir = ds / split / "images"
            lbl_dir = ds / split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                relabeled = relabel(lbl.read_text(), src_map)
                if not relabeled.strip():
                    continue  # no shahed boxes survived remap
                h = file_hash(img)
                if h in seen:
                    continue
                seen.add(h)
                rows.append((img, relabeled, ds.name, modality))
                per_ds += 1
        print(f"  {ds.name}: {per_ds:>5} kept  ({modality})")

    if not rows:
        raise SystemExit("no usable images found")

    rows.sort(key=lambda r: (r[2], r[0].name))
    cut = int(len(rows) * SPLIT_RATIO)
    splits = {"train": rows[:cut], "val": rows[cut:]}

    print(f"\ntotal: {len(rows)}  (train {len(splits['train'])} / val {len(splits['val'])})")
    by_modality = Counter(r[3] for r in rows)
    print(f"modality: {dict(by_modality)}")
    by_source = Counter(r[2] for r in rows)
    for src, n in by_source.most_common():
        print(f"  {n:>5}  {src}")

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in splits:
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)

    for split, items in splits.items():
        for i, (img, lbl_txt, slug, _mod) in enumerate(items):
            stem = f"{slug}_{i:06d}"
            shutil.copy(img, OUT / split / "images" / (stem + img.suffix.lower()))
            (OUT / split / "labels" / (stem + ".txt")).write_text(lbl_txt)

    data_yaml = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "names": {CLASS_ID: CLASS_NAME},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"\nwrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
