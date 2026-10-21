"""Fine-tune a COCO-pretrained YOLO11n on VisDrone person data (E1, RGB baseline).

CPU-friendly defaults: 1,000-image training subset, small epoch count.
On a GPU, raise --epochs and use the full train set
(python scripts/prepare_visdrone.py --train-subset 0).

Usage:
    python scripts/train.py --epochs 5 --name ft_yolo11n
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--data", default="data/visdrone_person/data.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="ft_yolo11n")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, seed=args.seed, deterministic=True, workers=2,
                single_cls=True, project="runs/train", name=args.name, exist_ok=True,
                val=True, plots=True)


if __name__ == "__main__":
    main()
