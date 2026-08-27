"""Extract labelled Arabic-letter crops from sample plates.

Usage:
    python -m anpr_maroc.scripts.extract_arabic_letter_dataset \
      --images data/sample_plates --output data/arabic_letters/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from anpr_maroc.processing.segmenter import segment_plate_by_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ground-truth Arabic letter crops.")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing plates and ground_truth.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory, grouped by Arabic letter")
    args = parser.parse_args()

    with (args.images / "ground_truth.json").open(encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    written = 0
    for filename, expected in ground_truth.items():
        image = cv2.imread(str(args.images / filename))
        if image is None:
            print(f"skip unreadable: {filename}")
            continue
        zones = segment_plate_by_layout(image)
        letter = str(expected["lettre"])
        target = args.output / letter
        target.mkdir(parents=True, exist_ok=True)
        output = target / f"{Path(filename).stem}.png"
        if not cv2.imwrite(str(output), zones[1]):
            raise OSError(f"could not write {output}")
        written += 1
    print(f"Extracted {written} labelled crops into {args.output}")


if __name__ == "__main__":
    main()
