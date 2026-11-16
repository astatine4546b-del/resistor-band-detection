# Resistor Color Band Detection

YOLO11n baseline for detecting resistor bodies and individual color bands. The model outputs bounding boxes and class labels.

| Test set (131 images) | Value |
| --- | ---: |
| Precision | 97.83% |
| Recall | 96.42% |
| mAP@50 | 96.42% |
| mAP@50–95 | 78.83% |

These metrics cover the annotated detection classes, including the resistor body and breadboard. Whole-resistor color sequence and resistance-value accuracy have yet to be evaluated.

- [Training metrics](runs/resistor-yolo11n-v6/results.csv)
- [Test metrics](runs/resistor-yolo11n-v6/test/metrics.yaml)
- [Best checkpoint](runs/resistor-yolo11n-v6/weights/best.pt)

## Reproduce the training

The run used Python 3.12, Ultralytics 8.4.208, and a CUDA-capable GPU. Download the YOLO export of [Resistor and Band Detection v6](https://universe.roboflow.com/jbhepner/resistor-and-band-detection/dataset/6) by jbhepner (CC BY 4.0), and extract it into `Resistor and Band Detection.v6i.yolov11/` beside the training script. Place the Ultralytics YOLO11n pretrained checkpoint, `yolo11n.pt`, in the repository root.

```sh
python -m pip install ultralytics==8.4.208
python train_resistor.py --audit-only
python train_resistor.py --name resistor-yolo11n-reproduction
```

The script converts polygon labels to bounding boxes, trains for 30 epochs by default, and evaluates the best checkpoint on the test split.
