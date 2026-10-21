"""Run inference and save annotated sample predictions (people only).

Usage:
    python scripts/predict.py --weights runs/train/ft_yolo11n/weights/best.pt \
        --source data/visdrone_person/val/images --n 8 --out results/samples/finetuned
"""
import argparse
import json
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="data/visdrone_person/val/images")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/samples")
    args = ap.parse_args()

    imgs = sorted(Path(args.source).glob("*.jpg"))
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    records = []
    for p in imgs:
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, classes=[0],
                            device="cpu", verbose=False)[0]
        img = res.orig_img.copy()
        dets = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{c:.2f}", (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1, cv2.LINE_AA)
            dets.append({"bbox_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)], "conf": round(c, 4)})
        label = f"{len(dets)} person detections (conf>={args.conf})"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / p.name), img)
        records.append({"image": p.name, "num_detections": len(dets), "detections": dets})
    (out / "predictions.json").write_text(json.dumps(records, indent=2))
    print(f"saved {len(records)} annotated images to {out}")


if __name__ == "__main__":
    main()
