"""Build an annotation queue of real Arabic-letter crops from the plate dataset.

``data/plate_detection`` holds 434 photographed Moroccan plates with
ground-truth plate boxes.  Cropping with those boxes and running the production
segmenter gives letter crops drawn from exactly the distribution the inference
path sees -- which is what the handwritten AHCD corpus cannot provide.

    python -m anpr_maroc.scripts.build_letter_queue
    python data/arabic_letters/annotator/serve_annotator.py   # then label them

Two properties of the Roboflow export drive the design:

 - Every image is one of six augmentations of a base scene; ``flip`` and
   ``flip_vert`` mirror the plate, and a mirrored Arabic glyph is a different
   shape than the letter it came from.  Only geometry-preserving variants are
   used, one per scene, so the queue holds each real plate exactly once.
 - The export splits augmentations, not scenes, so 348 of the 434 scenes have
   variants in more than one of train/valid/test.  Its splits are therefore
   unusable as a leakage boundary here and a fresh grouped split is needed.
   Scenes shot within ``--burst-gap`` seconds of each other are almost always
   the same car, so they are grouped and must not be split apart later.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
import typing as t
from pathlib import Path

import cv2
import numpy as np

from anpr_maroc.ocr.arabic_letter_model import preprocess_letter
from anpr_maroc.processing.segmenter import segment_plate_by_layout

# <scene>_<augmentation>_jpg.rf.<hash>.jpg
AUGMENTED_NAME = re.compile(
    r"^(?P<scene>.*)_(?P<aug>flip_vert|flip|trans|shear|contrast|noise)_jpg\.rf\.[A-Za-z0-9]+\.jpg$"
)
SCENE_TIMESTAMP = re.compile(r"^(\d{8})_(\d{6})")

# Mirroring is excluded outright; the rest are ordered by how little they
# disturb the glyph, so a scene contributes its cleanest available variant.
GEOMETRY_PRESERVING = ("contrast", "noise", "trans", "shear")

DEFAULT_DATASET = Path("data/plate_detection")
DEFAULT_OUTPUT = Path("data/arabic_letters/to_label")
MANIFEST_NAME = "queue_manifest.jsonl"


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian; low values mean motion blur or defocus."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_variants(dataset: Path) -> dict[str, tuple[str, str, Path]]:
    """Map each scene to one geometry-preserving (augmentation, split, path)."""
    found: dict[str, dict[str, tuple[str, Path]]] = collections.defaultdict(dict)
    for split in ("train", "valid", "test"):
        for path in sorted((dataset / split / "images").glob("*.jpg")):
            match = AUGMENTED_NAME.match(path.name)
            if match is None:
                print(f"[WARN] unrecognised name, skipped: {path.name}")
                continue
            found[match["scene"]][match["aug"]] = (split, path)

    chosen: dict[str, tuple[str, str, Path]] = {}
    for scene, variants in found.items():
        for aug in GEOMETRY_PRESERVING:
            if aug in variants:
                split, path = variants[aug]
                chosen[scene] = (aug, split, path)
                break
        else:
            print(f"[WARN] {scene}: only mirrored variants exist, skipped")
    return chosen


def largest_box(label_path: Path, width: int, height: int) -> t.Optional[tuple[int, int, int, int]]:
    """Read a YOLO label file and return the biggest box in pixel coordinates.

    A frame may catch a second plate in the background; the largest box is the
    plate the photograph was taken of.
    """
    if not label_path.is_file():
        return None
    best: t.Optional[tuple[float, float, float, float]] = None
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(value) for value in parts[1:5])
        if best is None or bw * bh > best[2] * best[3]:
            best = (cx, cy, bw, bh)
    if best is None:
        return None
    cx, cy, bw, bh = best
    x1 = max(0, int((cx - bw / 2) * width))
    y1 = max(0, int((cy - bh / 2) * height))
    x2 = min(width, int((cx + bw / 2) * width))
    y2 = min(height, int((cy + bh / 2) * height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def burst_groups(scenes: t.Iterable[str], gap_seconds: float) -> dict[str, str]:
    """Group scenes photographed within ``gap_seconds`` of each other.

    Consecutive frames of one car must not land on opposite sides of a
    train/test split, or the test score measures memorisation.  Grouping is
    deliberately generous: merging two cars costs a little data independence,
    while splitting one car invalidates the evaluation.
    """
    timed: list[tuple[datetime.datetime, str]] = []
    undated: list[str] = []
    for scene in scenes:
        match = SCENE_TIMESTAMP.match(scene)
        if match is None:
            undated.append(scene)
            continue
        timed.append((datetime.datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S"), scene))

    groups: dict[str, str] = {scene: f"solo_{scene}" for scene in undated}
    timed.sort()
    index = 0
    for position, (moment, scene) in enumerate(timed):
        if position > 0 and (moment - timed[position - 1][0]).total_seconds() > gap_seconds:
            index += 1
        groups[scene] = f"burst_{index:03d}"
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract real plate letter crops into the annotation queue."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Roboflow plate-detection dataset root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where letter crops are written.")
    parser.add_argument("--min-size", type=int, default=16, help="Reject crops whose smaller side is below this, in pixels.")
    parser.add_argument("--burst-gap", type=float, default=10.0, help="Seconds between shots below which two scenes are treated as the same car.")
    parser.add_argument("--keep-invalid-glyph", action="store_true", help="Queue crops even when preprocessing finds no isolated glyph.")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"dataset not found: {args.dataset}")

    chosen = select_variants(args.dataset)
    if not chosen:
        raise SystemExit(f"no usable images under {args.dataset}")
    groups = burst_groups(chosen, args.burst_gap)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.parent / MANIFEST_NAME
    counters = collections.Counter()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for scene, (aug, split, image_path) in sorted(chosen.items()):
            counters["scenes"] += 1
            image = cv2.imread(str(image_path))
            if image is None:
                counters["unreadable"] += 1
                continue
            height, width = image.shape[:2]
            box = largest_box(
                image_path.parent.parent / "labels" / f"{image_path.stem}.txt", width, height
            )
            if box is None:
                counters["no_box"] += 1
                continue
            x1, y1, x2, y2 = box
            plate = image[y1:y2, x1:x2]

            try:
                zones = segment_plate_by_layout(plate)
            except Exception as exc:  # segmentation is heuristic; never abort the run
                print(f"[WARN] {scene}: segmentation failed ({exc})")
                counters["no_segment"] += 1
                continue
            if len(zones) != 3 or zones[1] is None or zones[1].size == 0:
                counters["no_segment"] += 1
                continue
            letter = zones[1]

            if min(letter.shape[:2]) < args.min_size:
                counters["too_small"] += 1
                continue

            # The CNN's own preprocessing is the strictest validity test we have:
            # it fails when the zone holds no isolated glyph, which catches the
            # mis-segmented crops the annotator would otherwise have to reject.
            glyph_ok = True
            try:
                preprocess_letter(letter)
            except (ValueError, cv2.error):
                glyph_ok = False
                if not args.keep_invalid_glyph:
                    counters["no_glyph"] += 1
                    continue

            crop_path = args.output / f"{scene}.png"
            if not cv2.imwrite(str(crop_path), letter):
                raise OSError(f"could not write {crop_path}")

            manifest.write(json.dumps({
                "file": crop_path.name,
                "scene": scene,
                "group": groups[scene],
                "source_image": str(image_path),
                "source_split": split,   # recorded for provenance only: the export
                "augmentation": aug,     # splits augmentations, not scenes.
                "plate_box": [x1, y1, x2, y2],
                "size": [int(letter.shape[1]), int(letter.shape[0])],
                "sharpness": round(sharpness(letter), 1),
                "glyph_ok": glyph_ok,
            }, ensure_ascii=False) + "\n")
            counters["queued"] += 1

    print(f"[INFO] scenes considered : {counters['scenes']}")
    for key in ("unreadable", "no_box", "no_segment", "too_small", "no_glyph"):
        if counters[key]:
            print(f"[INFO] dropped {key:<12}: {counters[key]}")
    print(f"[INFO] queued            : {counters['queued']} -> {args.output}")
    print(f"[INFO] manifest          : {manifest_path}")
    print(f"[INFO] independent groups: {len(set(groups.values()))}")
    print(f"[INFO] label them with   : python {Path('data/arabic_letters/annotator/serve_annotator.py')}")


if __name__ == "__main__":
    main()
