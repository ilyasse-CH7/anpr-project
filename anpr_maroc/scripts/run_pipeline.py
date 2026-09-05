"""Simple CLI to run the full ANPR pipeline on one or multiple images and print concise CSV-like results.

Usage:
  python -m anpr_maroc.scripts.run_pipeline --image path/to/img.jpg
  python -m anpr_maroc.scripts.run_pipeline --dir path/to/images --segment

Produces one-line-per-image: filename,left,letter,right,valid,letter_conf,raw_text
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import cv2

from anpr_maroc.ocr.easyocr_reader import EasyOCRReader


def process_one(reader: EasyOCRReader, image_path: Path, segment: bool = False):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"{image_path},,, ,False,0.0,ERROR:unreadable")
        return
    try:
        res = reader.read_and_parse(img, try_segment=(reader.segment_wrapper if hasattr(reader,'segment_wrapper') else None), debug_dir=None) if segment else reader.read_and_parse(img)
    except Exception as e:
        print(f"{image_path},,, ,False,0.0,ERROR:{e}")
        return
    parsed = res.get('parsed', {})
    left = parsed.get('left','')
    letter = parsed.get('letter','')
    right = parsed.get('right','')
    valid = parsed.get('valid', False)
    # try to infer letter confidence from zone results
    letter_conf = 0.0
    for zr in res.get('zone_results', []):
        if zr.get('zone') == 'letter':
            letter_conf = float(zr.get('conf', 0.0))
    raw_text = res.get('raw_text','').replace(',', ' ')
    print(f"{image_path.name},{left},{letter},{right},{int(bool(valid))},{letter_conf:.3f},{raw_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, help='Single image path')
    parser.add_argument('--dir', type=Path, help='Directory of images to process')
    parser.add_argument('--segment', action='store_true', help='Enable segmentation-based processing (try segmenter)')
    args = parser.parse_args()

    reader = EasyOCRReader()
    # prefer segmentation path to avoid fallback to missing methods
    try:
        from anpr_maroc.processing.segmenter import segment_plate_by_layout
        seg_fn = segment_plate_by_layout
    except Exception:
        seg_fn = None

    # header
    print('filename,left,letter,right,valid,letter_conf,raw_text')
    if args.image:
        if seg_fn:
            res = reader.read_and_parse(args.image, try_segment=seg_fn)
            parsed = res.get('parsed', {})
            left = parsed.get('left','')
            letter = parsed.get('letter','')
            right = parsed.get('right','')
            valid = parsed.get('valid', False)
            letter_conf = 0.0
            for zr in res.get('zone_results', []):
                if zr.get('zone') == 'letter':
                    letter_conf = float(zr.get('conf', 0.0))
            raw_text = res.get('raw_text','').replace(',',' ')
            print(f"{args.image.name},{left},{letter},{right},{int(bool(valid))},{letter_conf:.3f},{raw_text}")
        else:
            process_one(reader, args.image, segment=args.segment)
    elif args.dir:
        for p in sorted(args.dir.glob('*')):
            if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                if seg_fn:
                    try:
                        res = reader.read_and_parse(str(p), try_segment=seg_fn)
                        parsed = res.get('parsed', {})
                        left = parsed.get('left','')
                        letter = parsed.get('letter','')
                        right = parsed.get('right','')
                        valid = parsed.get('valid', False)
                        letter_conf = 0.0
                        for zr in res.get('zone_results', []):
                            if zr.get('zone') == 'letter':
                                letter_conf = float(zr.get('conf', 0.0))
                        raw_text = res.get('raw_text','').replace(',',' ')
                        print(f"{p.name},{left},{letter},{right},{int(bool(valid))},{letter_conf:.3f},{raw_text}")
                    except Exception as e:
                        print(f"{p.name},,, ,False,0.0,ERROR:{e}")
                else:
                    process_one(reader, p, segment=args.segment)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
