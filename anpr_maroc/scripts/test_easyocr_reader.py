"""Quick OCR smoke test on a local image.

Usage:
    python -m anpr_maroc.scripts.test_easyocr_reader --image data/sample_plates/2.jpg
    python -m anpr_maroc.scripts.test_easyocr_reader --image data/sample_plates/2.jpg --segment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import tempfile

from anpr_maroc.ocr.easyocr_reader import EasyOCRReader
from anpr_maroc.processing.segmenter import segment_plate_by_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EasyOCRReader on a plate image.")
    parser.add_argument("--image", type=str, required=True, help="Path to a sample plate image")
    parser.add_argument("--gpu", action="store_true", help="Enable CUDA/GPU mode")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum confidence threshold")
    parser.add_argument("--segment", action="store_true", help="Use 3-zone segmentation before OCR")
    parser.add_argument("--debug", action="store_true", help="Save segmented zone images to a temp folder for inspection")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = EasyOCRReader(gpu=args.gpu, conf_threshold=args.threshold)

    # load ground truth if present
    gt_path = image_path.parent / "ground_truth.json"
    ground_truth = {}
    if gt_path.exists():
        import json

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    def run_one(path: Path, debug_dir: str | None = None):
        plate = cv2.imread(str(path))
        if plate is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if debug_dir:
            print('Debug crops saved to', debug_dir)
        def seg_wrapper(img, debug_dir=None):
            return segment_plate_by_layout(img, debug_dir=debug_dir)
        result = reader.read_and_parse(plate, try_segment=seg_wrapper if args.segment else None, debug_dir=debug_dir)
        return result

    # run on requested image first (with debug if requested)
    debug_dir = None
    if args.debug and args.segment:
        debug_dir = str(Path(tempfile.mkdtemp(prefix='anpr_seg_')))
    current_result = run_one(image_path, debug_dir=debug_dir)

    print("RESULT for:", image_path.name)
    print("RAW_TEXT:", current_result.get("raw_text", ""))
    parsed = current_result.get("parsed") or current_result.get("parsed", {})
    print("PARSED:", parsed)
    if "zone_results" in current_result:
        print("\nBY ZONE:")
        for item in current_result["zone_results"]:
            print(item)
    if debug_dir:
        print('Saved segment images in', debug_dir)

    # compare with ground truth for this image if available
    if ground_truth and image_path.name in ground_truth:
        expected = ground_truth[image_path.name]
        ok_fields = {}
        ok = True
        for fld, key in [("serie", "left"), ("lettre", "letter"), ("region", "right")]:
            exp = str(expected[fld]) if fld in expected else ""
            got = str(parsed.get(key, "")).strip()
            ok_fields[key] = {"expected": exp, "got": got, "match": exp == got}
            if exp != got:
                ok = False
        status = "CORRECT" if ok else "INCORRECT"
        print(f"\nGROUND TRUTH CHECK: {status}")
        for k, v in ok_fields.items():
            print(f" - {k}: expected='{v['expected']}' got='{v['got']}' match={v['match']}")

    # if ground truth exists, run batch evaluation across all ground truth entries to compute accuracy
    if ground_truth:
        print('\nRunning batch evaluation across ground truth set...')
        total = 0
        correct_total = 0
        correct_field_counts = {"left": 0, "letter": 0, "right": 0}
        for fname, exp in ground_truth.items():
            p = image_path.parent / fname
            if not p.exists():
                print(' - skip missing:', fname)
                continue
            res = run_one(p, debug_dir=None)
            parsed_r = res.get('parsed', {})
            total += 1
            all_match = True
            for fld, key in [("serie", "left"), ("lettre", "letter"), ("region", "right")]:
                expected_val = str(exp[fld])
                got_val = str(parsed_r.get(key, "")).strip()
                if expected_val == got_val:
                    correct_field_counts[key] += 1
                else:
                    all_match = False
            if all_match:
                correct_total += 1
        if total > 0:
            overall_acc = correct_total / total * 100.0
            print(f"Batch accuracy: {correct_total}/{total} images exact match ({overall_acc:.1f}%)")
            for key, cnt in correct_field_counts.items():
                print(f" Field {key}: {cnt}/{total} correct ({cnt/total*100.0:.1f}%)")
        else:
            print('No ground truth images found for batch evaluation.')


if __name__ == "__main__":
    main()
