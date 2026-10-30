"""Extract people from an image using YOLO.

Detected people whose bounding boxes overlap are merged into a single crop,
so a group of overlapping people is written out as one picture. The input may
also be a directory: it is scanned recursively and the output directory mirrors
its layout.

Usage:
    python main.py <image-or-dir> <output-dir> [--model yolo26s.pt] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(source: Path, output_dir: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image, destination directory) pairs for an image or directory.

    Directories are scanned recursively and the destination mirrors their
    layout inside output_dir. The output tree is skipped when it lives inside
    the input tree so crops from an earlier run are not picked up again.
    """
    if not source.is_dir():
        yield source, output_dir
        return

    source_root = source.resolve()
    output_root = output_dir.resolve()
    skip = output_root if source_root in output_root.parents else None

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if skip is not None:
            resolved = path.resolve()
            if resolved == skip or skip in resolved.parents:
                continue
        yield path, output_dir / path.relative_to(source).parent


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


def process_image(
    image_path: Path,
    output_dir: Path,
    model: YOLO,
    conf: float,
    iou: float,
) -> tuple[int, int]:
    """Detect people in one image and save merged crops into output_dir.

    Returns (detected people, saved crops). Unreadable images are reported and
    treated as empty so a whole directory can still be processed.
    """
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - continue with the other images
        print(f"Skipping {image_path}: cannot read image ({exc})", file=sys.stderr)
        return 0, 0

    width, height = image.size

    results = model.predict(str(image_path), classes=[0], conf=conf, verbose=False)

    boxes = np.empty((0, 4), dtype=np.float32)
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)

    if len(boxes) == 0:
        print(f"No people detected in {image_path}.")
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0

    groups = union_find_groups(boxes, iou)

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

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"

    digits = max(3, len(str(len(crops))))
    prefix = image_path.stem
    saved = 0
    for index, (x1, y1, x2, y2) in enumerate(crops, start=1):
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        if right <= left or bottom <= top:
            continue

        crop = image.crop((left, top, right, bottom))
        out_path = output_dir / f"{prefix}_p{index:0{digits}d}{suffix}"
        crop.save(out_path)
        saved += 1

    print(f"Detected {len(boxes)} person(s) in {saved} crop(s).")
    return len(boxes), saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract people from an image, or from every image in a directory; "
            "overlapping people are saved as one crop."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an input image, or a directory scanned recursively",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for extracted people (mirrors the input directory layout)",
    )
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

    if not args.input.exists():
        sys.exit(f"error: input not found: {args.input}")
    if not (args.input.is_file() or args.input.is_dir()):
        sys.exit(f"error: not an image or directory: {args.input}")

    model: YOLO | None = None
    processed = 0
    total_people = 0
    total_crops = 0
    for image_path, target_dir in iter_images(args.input, args.output_dir):
        if model is None:
            model = YOLO(resolve_model(args.model))
        print(f"{image_path}:")
        people, crops = process_image(
            image_path, target_dir, model, args.conf, args.iou
        )
        processed += 1
        total_people += people
        total_crops += crops

    if processed == 0:
        print(f"No images found in {args.input}.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif args.input.is_dir():
        print(
            f"Processed {processed} image(s): "
            f"{total_people} person(s) in {total_crops} crop(s)."
        )


if __name__ == "__main__":
    main()
