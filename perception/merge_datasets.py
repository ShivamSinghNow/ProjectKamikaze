"""Merge data/raw/* YOLO datasets into data/merged with a single shahed class.

Walks each raw dataset, copies images + relabeled YOLO txt files to the merged
tree, hashes images to drop exact dupes, and emits a YOLO data.yaml.
"""
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "merged"

CLASS_ID = 0
CLASS_NAME = "shahed"
SPLIT_RATIO = 0.85  # train/val


def yolo_relabel(label_text: str) -> str:
    out = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        out.append(" ".join([str(CLASS_ID), *parts[1:5]]))
    return "\n".join(out)


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} — run download_datasets.py first")

    seen: set[str] = set()
    rows = []  # (src_img, src_lbl, dataset_slug)

    for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
        for split in ("train", "valid", "test"):
            img_dir = ds / split / "images"
            lbl_dir = ds / split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                h = file_hash(img)
                if h in seen:
                    continue
                seen.add(h)
                rows.append((img, lbl, ds.name))

    if not rows:
        raise SystemExit("no usable images found in data/raw")

    rows.sort()
    cut = int(len(rows) * SPLIT_RATIO)
    splits = {"train": rows[:cut], "val": rows[cut:]}

    by_source = Counter(r[2] for r in rows)
    print(f"unique images: {len(rows)}  (train {len(splits['train'])} / val {len(splits['val'])})")
    for src, n in by_source.most_common():
        print(f"  {n:>4}  {src}")

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in splits:
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)

    for split, items in splits.items():
        for i, (img, lbl, slug) in enumerate(items):
            stem = f"{slug}_{i:05d}"
            shutil.copy(img, OUT / split / "images" / (stem + img.suffix.lower()))
            (OUT / split / "labels" / (stem + ".txt")).write_text(yolo_relabel(lbl.read_text()))

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
