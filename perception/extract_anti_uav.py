"""Extract frames from Anti-UAV videos and convert annotations to YOLO format.

Usage:
    uv run python perception/extract_anti_uav.py \
        --src /mnt/ssd/antiuav/raw \
        --dst-rgb data/raw/antiuav__v300_rgb \
        --dst-ir  data/raw/antiuav__v300_ir \
        --stride 10 --val-split 0.15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

CLASS_ID = 0

RGB_VIDEO_NAMES = ("RGB.mp4", "visible.mp4")
RGB_LABEL_NAMES = ("RGB_label.json", "visible.json")
IR_VIDEO_NAMES = ("IR.mp4", "infrared.mp4")
IR_LABEL_NAMES = ("IR_label.json", "infrared.json")


def first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / names[0]


def process_video(video_path: Path, label_path: Path, out_root: Path, split: str, stride: int) -> int:
    if not video_path.exists() or not label_path.exists():
        return 0

    data = json.loads(label_path.read_text())
    exist = data.get("exist", [])
    rects = data.get("gt_rect", [])
    if not rects:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    seq = video_path.parent.name
    written = 0
    idx = 0
    while idx < len(rects):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0 and idx < len(exist) and exist[idx]:
            x, y, bw, bh = rects[idx]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if 0 < cx < 1 and 0 < cy < 1 and nw > 0 and nh > 0:
                stem = f"{seq}_{idx:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
                (lbl_dir / f"{stem}.txt").write_text(
                    f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                )
                written += 1
        idx += 1

    cap.release()
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst-rgb", required=True, type=Path)
    p.add_argument("--dst-ir", required=True, type=Path)
    p.add_argument("--stride", type=int, default=10, help="keep every Nth frame")
    p.add_argument("--val-split", type=float, default=0.15)
    args = p.parse_args()

    if (args.src / "train").exists() or (args.src / "val").exists():
        split_roots = (("train", args.src / "train"), ("val", args.src / "val"))
        seqs = [
            (split, seq)
            for split, root in split_roots
            if root.exists()
            for seq in sorted(root.iterdir())
            if seq.is_dir()
        ]
    else:
        dirs = sorted([d for d in args.src.iterdir() if d.is_dir()])
        random.seed(42)
        random.shuffle(dirs)
        cut = int(len(dirs) * (1 - args.val_split))
        seqs = [("train" if i < cut else "val", seq) for i, seq in enumerate(dirs)]

    rgb_total = ir_total = 0
    for i, (split, seq) in enumerate(seqs):
        rgb_total += process_video(
            first_existing(seq, RGB_VIDEO_NAMES),
            first_existing(seq, RGB_LABEL_NAMES),
            args.dst_rgb,
            split,
            args.stride,
        )
        ir_total += process_video(
            first_existing(seq, IR_VIDEO_NAMES),
            first_existing(seq, IR_LABEL_NAMES),
            args.dst_ir,
            split,
            args.stride,
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(seqs)}  rgb={rgb_total} ir={ir_total}")

    print(f"\ndone: {rgb_total} RGB frames, {ir_total} IR frames")


if __name__ == "__main__":
    main()
