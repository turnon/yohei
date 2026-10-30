# yohei

Extract people from an image with YOLO. Each detected person is saved as a separate crop. Detections that overlap are merged into a single crop, so a group of people standing together becomes one picture. The input can also be a directory, which is scanned recursively; the output directory mirrors its layout.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Model weights (`yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt`, `yolo26s.pt`) are downloaded automatically on first use into `models/` next to `main.py`. You can also drop weight files there yourself to skip the download. Weight files are git-ignored.

## Usage

```bash
uv run main.py <image-or-dir> <output-dir> [--model MODEL] [--conf CONF] [--iou IOU]
```

### Arguments

| Argument       | Required | Default       | Description                                                        |
| -------------- | -------- | ------------- | ------------------------------------------------------------------ |
| `input`        | yes      | —             | Input image, or a directory that is scanned recursively for images |
| `output_dir`   | yes      | —             | Directory to write the crops into (created if missing); mirrors the input layout for directory input |
| `--model`      | no       | `yolo26s.pt`  | YOLO weights to use; bare names resolve to `models/` next to `main.py` |
| `--conf`       | no       | `0.25`        | Detection confidence threshold                                     |
| `--iou`        | no       | `0.0`         | Minimum IoU for merging two detections into one crop               |

### Examples

Basic usage:

```bash
uv run main.py photo.jpg out/
```

Process every image in a directory tree, mirroring the subdirectories under `out/`:

```bash
uv run main.py photos/ out/
```

Use a smaller, faster model:

```bash
uv run main.py photo.jpg out/ --model yolo11n.pt
```

Be stricter about detections and only merge strongly overlapping people:

```bash
uv run main.py group.jpg out/ --conf 0.5 --iou 0.3
```

Use weights from a specific location (any path that includes a directory is used as given):

```bash
uv run main.py photo.jpg out/ --model ./tmp/custom.pt
```

Keep every person separate (merge only if boxes are identical):

```bash
uv run main.py group.jpg out/ --iou 1.0
```

## Model weights

Weights are cached in `models/` next to `main.py`, no matter which directory you run the command from:

| `--model` value           | Resolved path                              |
| ------------------------- | ------------------------------------------ |
| `yolo26s.pt` (bare name)  | `<project>/models/yolo26s.pt`, downloaded  |
| `./tmp/custom.pt`         | `<project>/tmp/custom.pt`, used as given   |
| `/abs/path/custom.pt`     | `/abs/path/custom.pt`, used as given       |
| `https://…/custom.pt`     | `<project>/models/custom.pt`, downloaded   |

Missing weights are fetched from the Ultralytics release assets on first use. Older weight files dumped in the project root can simply be moved into `models/` to skip the re-download.

## Output

Crops are written as `<original-name>_p001.<ext>`, `<original-name>_p002.<ext>`, … in left-to-right, top-to-bottom order, where `<original-name>` is the input image's file name without extension. This keeps results from different source images distinguishable when they share an output directory and avoids overwriting the input if it lives there. The original image's extension is preserved when supported (`.jpg`, `.png`, `.webp`, `.bmp`, `.tif`); otherwise `.png` is used.

For directory input, each image's relative directory is recreated under `output_dir`. Given `photos/trip/day1/a.jpg` and `photos/trip/day2/b.jpg`, the result looks like:

```
out/
└── trip/
    ├── day1/
    │   └── a_p001.jpg
    └── day2/
        └── b_p001.jpg
```

A previously generated output directory is skipped when it sits inside the input tree, so re-running does not reprocess old crops. Supported input extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff` (case-insensitive).

Example console output:

```
photos/trip/day1/a.jpg:
Saved out/trip/day1/a_p001.jpg (220x640)
Saved out/trip/day1/a_p002.jpg (410x660)
Detected 3 person(s) in 2 crop(s).
photos/trip/day2/b.jpg:
No people detected in photos/trip/day2/b.jpg.
Processed 2 image(s): 3 person(s) in 2 crop(s).
```

If no people are found in a single image, the output directory is still created and `No people detected in <image>.` is printed. If a directory contains no images, `No images found in <dir>.` is printed and the output directory is created.
