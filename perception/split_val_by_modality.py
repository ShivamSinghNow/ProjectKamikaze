"""Build YOLO data.yaml files for RGB-only and IR-only validation splits."""
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"

IR_KEYWORDS = ("ir", "thermal", "infrared")


def is_ir(filename: str) -> bool:
    name = filename.lower()
    return any(keyword in name for keyword in IR_KEYWORDS)


for modality in ("rgb", "ir"):
    out = ROOT / "data" / f"merged_{modality}_val"
    if out.exists():
        shutil.rmtree(out)
    (out / "val" / "images").mkdir(parents=True)
    (out / "val" / "labels").mkdir(parents=True)

    src_imgs = MERGED / "val" / "images"
    src_lbls = MERGED / "val" / "labels"
    n = 0
    for img in src_imgs.iterdir():
        target_ir = is_ir(img.name)
        if (modality == "ir") != target_ir:
            continue
        shutil.copy(img, out / "val" / "images" / img.name)
        lbl = src_lbls / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, out / "val" / "labels" / lbl.name)
        n += 1

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "val/images",
                "val": "val/images",
                "names": {0: "shahed"},
            }
        )
    )
    print(f"{modality}: {n} images at {out / 'data.yaml'}")
