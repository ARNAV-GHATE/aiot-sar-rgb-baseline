"""Build the results table and figures from saved metrics / runs.

Outputs (results/):
  results_table.md           - zero-shot vs fine-tuned metrics
  training_curves.png        - val mAP@0.5 and recall per epoch (from Ultralytics results.csv)
  comparison_<img>.jpg       - same val image, zero-shot (left) vs fine-tuned (right)
"""
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
RUN_CSV = Path("runs/detect/runs/train/ft_yolo11n/results.csv")


def table():
    rows = [json.loads((R / f"metrics_{n}.json").read_text())
            for n in ("zeroshot_yolo11n", "finetuned_yolo11n")]
    labels = {"zeroshot_yolo11n": "YOLO11n, COCO-pretrained (zero-shot)",
              "finetuned_yolo11n": "YOLO11n, fine-tuned 10 epochs on 1,000 VisDrone images"}
    lines = ["| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | False-negative rate | CPU latency (ms/img, mean / p95) |",
             "|---|---|---|---|---|---|---|"]
    for m in rows:
        lines.append(f"| {labels[m['name']]} | {m['mAP50']:.3f} | {m['mAP50_95']:.3f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['false_negative_rate']:.3f} | "
                     f"{m['latency_ms_mean']:.1f} / {m['latency_ms_p95']:.1f} |")
    lines += ["", f"Evaluation set: {rows[0]['dataset']}; imgsz {rows[0]['imgsz']}.",
              f"Hardware: {rows[0]['hardware']}. Software: {rows[0]['software']}.",
              "Precision/recall are at the confidence threshold that maximises F1 (Ultralytics default)."]
    (R / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def curves():
    with RUN_CSV.open() as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    ep = [int(float(r["epoch"])) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    ax.plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], marker="o", label="mAP@0.5")
    ax.plot(ep, [float(r["metrics/recall(B)"]) for r in rows], marker="s", label="Recall")
    ax.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in rows], marker="^", label="mAP@0.5:0.95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_title("YOLO11n fine-tuning on VisDrone (person), CPU")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "training_curves.png")


def comparisons(k=3):
    zs, ft = R / "samples/zeroshot", R / "samples/finetuned"
    for p in sorted(zs.glob("*.jpg"))[:k]:
        a, b = cv2.imread(str(p)), cv2.imread(str(ft / p.name))
        h = min(a.shape[0], b.shape[0])
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        for img, t in ((a, "Zero-shot (COCO)"), (b, "Fine-tuned (VisDrone)")):
            cv2.putText(img, t, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, t, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        sep = 255 * (a[:, :8] * 0 + 1)
        cv2.imwrite(str(R / f"comparison_{p.stem}.jpg"), cv2.hconcat([a, sep.astype(a.dtype), b]))


if __name__ == "__main__":
    table()
    curves()
    comparisons()
