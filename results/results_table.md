| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | False-negative rate | CPU latency (ms/img, mean / p95) |
|---|---|---|---|---|---|---|
| YOLO11n, COCO-pretrained (zero-shot) | 0.174 | 0.060 | 0.486 | 0.180 | 0.820 | 79.1 / 102.1 |
| YOLO11n, fine-tuned 10 epochs on 1,000 VisDrone images | 0.308 | 0.101 | 0.491 | 0.312 | 0.688 | 74.9 / 97.3 |

Evaluation set: VisDrone2019-DET-val (548 images, pedestrian+people -> person); imgsz 640.
Hardware: CPU only, 2 vCPUs, x86_64, no GPU. Software: python 3.13.16, torch 2.14.1+cu130, ultralytics 8.4.183.
Precision/recall are at the confidence threshold that maximises F1 (Ultralytics default).
