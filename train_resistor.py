import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import os
import shutil


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Resistor and Band Detection.v6i.yolov11"
PREPARED = ROOT / "resistor-detection-v6"
DATA_CONFIG = PREPARED / "data.yaml"
PROJECT = ROOT / "runs"


def prepare_dataset(class_names, audit_only):
    sources = {}
    for split in ("train", "valid", "test"):
        images = {p.stem: p for p in (DATASET / split / "images").glob("*.jpg")}
        labels = {p.stem: p for p in (DATASET / split / "labels").glob("*.txt")}
        if not images or images.keys() != labels.keys():
            raise ValueError(f"{split}: unmatched images or labels")
        sources[split] = {stem.split(".rf.", 1)[0] for stem in images}
        counts = Counter()
        polygons = 0
        out = PREPARED / split
        if not audit_only:
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "labels").mkdir(exist_ok=True)
        for label in labels.values():
            converted = []
            for row in label.read_text(encoding="utf-8").splitlines():
                class_id, *box = row.split()
                class_id = int(class_id)
                coords = list(map(float, box))
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    if not all(0 <= coord <= 1 for coord in coords):
                        raise ValueError(f"{label}: out-of-range polygon {row}")
                    xs, ys = coords[::2], coords[1::2]
                    coords = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                              max(xs) - min(xs), max(ys) - min(ys)]
                    polygons += 1
                x, y, width, height = coords
                if not (
                    0 <= class_id < len(class_names)
                    and 0 <= x <= 1 and 0 <= y <= 1
                    and 0 < width <= 1 and 0 < height <= 1
                ):
                    raise ValueError(f"{label}: invalid label {row}")
                counts[class_names[class_id]] += 1
                converted.append(f"{class_id} " + " ".join(f"{v:.9f}" for v in coords))
            if not audit_only:
                image = images[label.stem]
                copy = out / "images" / image.name
                if not copy.exists() or copy.stat().st_size != image.stat().st_size:
                    shutil.copyfile(image, copy)
                (out / "labels" / label.name).write_text("\n".join(converted), encoding="utf-8")
        print(f"{split}: {len(images)} images, {sum(counts.values())} boxes, {polygons} polygons")
        print(dict(sorted(counts.items())))
    for left, right in combinations(sources, 2):
        overlap = sources[left] & sources[right]
        if overlap:
            raise ValueError(f"{left}/{right} share {len(overlap)} original image names")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="resistor-yolo11n-v6")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="Path to an interrupted run's last.pt")
    args = parser.parse_args()

    import yaml

    data = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    prepare_dataset(data["names"], args.audit_only)
    if args.audit_only:
        return

    data.update(path=PREPARED.as_posix(), train="train/images",
                val="valid/images", test="test/images")
    DATA_CONFIG.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    if args.prepare_only:
        return
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".resistor-yolo-env" / "Ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".resistor-yolo-env"))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True, device=0, batch=args.batch)
    else:
        model = YOLO(ROOT / "yolo11n.pt")
        model.train(
            data=str(DATA_CONFIG), project=str(PROJECT), name=args.name,
            device=0, epochs=args.epochs, imgsz=640, batch=args.batch,
            workers=args.workers, patience=10, seed=42, save_period=5,
            hsv_h=0.0, hsv_s=0.1, hsv_v=0.1, fliplr=0.0, flipud=0.0,
            mosaic=0.0, scale=0.1, translate=0.05, amp=True,
        )
    run_dir = model.trainer.save_dir
    model = YOLO(run_dir / "weights" / "best.pt")
    metrics = model.val(
        data=str(DATA_CONFIG), split="test", device=0, imgsz=640,
        batch=args.batch, workers=args.workers, plots=True,
        project=str(run_dir), name="test",
    )
    (metrics.save_dir / "metrics.yaml").write_text(
        yaml.safe_dump({key: float(value) for key, value in metrics.results_dict.items()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
