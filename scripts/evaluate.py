"""Evaluate a YOLO detector on the VisDrone person val split (E1, RGB baseline).

Works for both:
  * a COCO-pretrained model used zero-shot (person = COCO class 0), and
  * a model fine-tuned on data/visdrone_person (single class 0 = person).
In both cases only class 0 is scored, so the comparison is like-for-like.

Reports mAP@0.5, mAP@0.5:0.95, precision, recall, false-negative rate
(1 - recall at the operating point), and mean per-image inference latency
measured separately on CPU.

Usage:
    python scripts/evaluate.py --weights yolo11n.pt --name zeroshot_yolo11n
    python scripts/evaluate.py --weights runs/train/ft_yolo11n/weights/best.pt --name finetuned_yolo11n
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def measure_latency(model, img_dir: Path, imgsz: int, n: int = 50, warmup: int = 5) -> dict:
    imgs = sorted(img_dir.glob("*.jpg"))[:n]
    for p in imgs[:warmup]:
        model.predict(str(p), imgsz=imgsz, classes=[0], device="cpu", verbose=False)
    times = []
    for p in imgs:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, classes=[0], device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_mean": float(np.mean(times)), "latency_ms_p95": float(np.percentile(times, 95)),
            "latency_images": len(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/visdrone_person/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=8, device="cpu", classes=[0],
                        conf=0.001, iou=0.6, max_det=1000, project="runs/val", name=args.name, exist_ok=True,
                        plots=True, verbose=False)
    p, r, map50, map5095 = metrics.box.mean_results()

    val_dir = Path(args.data).parent / "val" / "images"
    lat = measure_latency(model, val_dir, args.imgsz)

    result = {
        "name": args.name,
        "weights": args.weights,
        "dataset": "VisDrone2019-DET-val (548 images, pedestrian+people -> person)",
        "imgsz": args.imgsz,
        "precision": float(p),
        "recall": float(r),
        "false_negative_rate": float(1 - r),
        "mAP50": float(map50),
        "mAP50_95": float(map5095),
        **lat,
        "hardware": f"CPU only, {os.cpu_count()} vCPUs, {platform.machine()}, no GPU",
        "software": f"python {platform.python_version()}, torch {torch.__version__}, ultralytics {ultralytics.__version__}",
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"metrics_{args.name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
