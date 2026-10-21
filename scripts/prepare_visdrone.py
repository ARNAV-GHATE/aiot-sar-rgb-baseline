"""Convert VisDrone2019-DET to a single-class ("person") YOLO dataset.

VisDrone annotation row: x,y,w,h,score,category,truncation,occlusion
Categories: 0 ignored-region, 1 pedestrian, 2 people, 3 bicycle, 4 car, ...
For the SAR person-detection baseline, categories 1 (pedestrian) and 2
(people = sitting/non-standing person) are merged into class 0 "person".
Boxes with score==0 (ignored) are dropped. Ignored regions are left
unlabelled, as is standard for YOLO-format conversions of VisDrone.

Usage:
    python scripts/prepare_visdrone.py --raw data/raw --out data/visdrone_person \
        --train-subset 1000 --seed 0
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

PERSON_CATS = {1, 2}


def convert_split(src: Path, dst: Path, names: list[str]) -> tuple[int, int]:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for stem in names:
        img_path = src / "images" / f"{stem}.jpg"
        w_img, h_img = Image.open(img_path).size
        lines = []
        for row in (src / "annotations" / f"{stem}.txt").read_text().strip().splitlines():
            vals = [int(v) for v in row.strip().strip(",").split(",")[:8]]
            x, y, w, h, score, cat = vals[:6]
            if score == 0 or cat not in PERSON_CATS or w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
        n_boxes += len(lines)
        (dst / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        target = dst / "images" / f"{stem}.jpg"
        if not target.exists():
            target.symlink_to(img_path.resolve())
    return len(names), n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/visdrone_person")
    ap.add_argument("--train-subset", type=int, default=1000,
                    help="number of training images to sample (0 = all 6471)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    train_src = raw / "VisDrone2019-DET-train"
    val_src = raw / "VisDrone2019-DET-val"
    train_names = sorted(p.stem for p in (train_src / "images").glob("*.jpg"))
    val_names = sorted(p.stem for p in (val_src / "images").glob("*.jpg"))

    if args.train_subset:
        # Random image subset with a fixed seed (CPU-only compute budget).
        # The official VisDrone train/val split is kept, so val scenes stay unseen.
        rng = random.Random(args.seed)
        rng.shuffle(train_names)
        train_names = sorted(train_names[: args.train_subset])

    n_tr, b_tr = convert_split(train_src, out / "train", train_names)
    n_va, b_va = convert_split(val_src, out / "val", val_names)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n"
    )
    (out / "train_list.txt").write_text("\n".join(train_names) + "\n")
    print(f"train: {n_tr} images, {b_tr} person boxes")
    print(f"val:   {n_va} images, {b_va} person boxes")


if __name__ == "__main__":
    main()
