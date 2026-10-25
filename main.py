"""Extract people from an image using YOLO.

Detected people whose bounding boxes overlap are merged into a single crop,
so a group of overlapping people is written out as one picture.

Usage:
    python main.py <image> <output-dir> [--model yolo26s.pt] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def resolve_model(spec: str) -> str:
    """Map a model spec to a local weights file, keeping downloads in MODELS_DIR.

    - bare names (`yolo26s.pt`) live in `<project>/models/` and are downloaded there
    - relative paths (`./w.pt`, `weights/w.pt`) are resolved against the project dir
    - absolute paths are used as-is
    - URLs are downloaded into `<project>/models/`
    """
    if spec.startswith(("http://", "https://")):
        from ultralytics.utils.downloads import safe_download

        name = Path(unquote(urlparse(spec).path)).name or "model.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return str(safe_download(url=spec, file=MODELS_DIR / name))

    path = Path(spec).expanduser()
    if path.is_absolute():
        target = path
    elif len(path.parts) > 1:  # explicit relative path, e.g. weights/custom.pt
        target = BASE_DIR / path
    else:  # bare filename -> models/ next to main.py
        target = MODELS_DIR / path.name

    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def union_find_groups(boxes: np.ndarray, iou_threshold: float) -> list[list[int]]:
    """Group box indices that overlap (IoU above threshold) transitively."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union_area = area_a + area_b - inter
        return inter / union_area if union_area > 0 else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) >= iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract people from an image; overlapping people are saved as one crop."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("output_dir", type=Path, help="Directory for extracted people")
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="YOLO model to use; bare names are downloaded into models/ (default: yolo26s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.0,
        help="Minimum IoU for merging detections into one crop (default: 0.0, i.e. any overlap)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        sys.exit(f"error: image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    model = YOLO(resolve_model(args.model))
    results = model.predict(str(args.image), classes=[0], conf=args.conf, verbose=False)

    boxes = np.empty((0, 4), dtype=np.float32)
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)

    if len(boxes) == 0:
        print("No people detected.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return

    groups = union_find_groups(boxes, args.iou)

    # Merge each group into one crop rectangle.
    crops: list[tuple[float, float, float, float]] = []
    for group in groups:
        merged = boxes[group]
        crops.append(
            (
                float(merged[:, 0].min()),
                float(merged[:, 1].min()),
                float(merged[:, 2].max()),
                float(merged[:, 3].max()),
            )
        )

    # Stable, human-friendly ordering: left to right, then top to bottom.
    crops.sort(key=lambda b: (round(b[0]), round(b[1])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.image.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"

    digits = max(3, len(str(len(crops))))
    for index, (x1, y1, x2, y2) in enumerate(crops, start=1):
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        if right <= left or bottom <= top:
            continue

        crop = image.crop((left, top, right, bottom))
        out_path = args.output_dir / f"p{index:0{digits}d}{suffix}"
        crop.save(out_path)
        print(f"Saved {out_path} ({crop.width}x{crop.height})")

    print(f"Detected {len(boxes)} person(s) in {len(crops)} crop(s).")


if __name__ == "__main__":
    main()
