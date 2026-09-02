"""Live camera capture for ANPR pipeline.

Usage examples:
  source .venv/bin/activate
  export PYTHONPATH=$PWD
  python -m anpr_maroc.scripts.live_camera --source 0
  python -m anpr_maroc.scripts.live_camera --source rtsp://192.168.1.10:554/stream --interval 0.5 --post-url http://localhost:8000/api/anpr

Behavior:
 - Grab frames from camera or video source
 - Run YOLO detection -> segmentation -> OCR (+ Arabic CNN)
 - Print concise, clear lines in terminal for each detected matricule: timestamp, left|letter|right, validity, confidences
 - Debounce repeated detections (cooldown configurable)
 - Optionally POST JSON results to backend URL (--post-url or ANPR_BACKEND_URL env var)
 - Optionally save detected plate crops to data/pipeline_output/live/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import json

import cv2

from anpr_maroc.detection.plate_detector import detect_plate
from anpr_maroc.processing.segmenter import segment_plate_by_layout
from anpr_maroc.ocr.easyocr_reader import EasyOCRReader

try:
    import requests
except Exception:
    requests = None


def iso_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def save_crop(img, out_dir: Path, prefix: str = "plate") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    p = out_dir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(p), img)
    return p


def process_frame(
    frame,
    reader: EasyOCRReader,
    model_path: str = "models/plate_detector.pt",
    conf_threshold: float = 0.25,
    save_dir: Path | None = None,
    post_url: str | None = None,
):
    # 1. detect plate
    bbox = None
    try:
        bbox = detect_plate(frame, model_path=model_path, conf_threshold=conf_threshold)
    except Exception as e:
        # detection error; treat as no bbox
        print(f"[WARN] detect_plate error: {e}")
        bbox = None

    is_already_cropped = False
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        plate_crop = frame[y1:y2, x1:x2]
    else:
        plate_crop = frame
        is_already_cropped = True

    # 2. run OCR + segmentation
    try:
        ocr_result = reader.read_and_parse(plate_crop, try_segment=lambda img: segment_plate_by_layout(img))
    except Exception as e:
        print(f"[ERROR] read_and_parse failed: {e}")
        return None

    parsed = ocr_result.get("parsed", {})
    serie = parsed.get("left", "")
    letter = parsed.get("letter", "")
    region = parsed.get("right", "")
    valid = bool(parsed.get("valid", False))

    # estimate letter confidence from zone_results
    letter_conf = 0.0
    for zr in ocr_result.get("zone_results", []) if ocr_result.get("zone_results") else []:
        if zr.get("zone") == "letter":
            letter_conf = float(zr.get("conf", 0.0))

    raw = ocr_result.get("raw_text", "")

    result = {
        "timestamp": iso_ts(),
        "matricule": f"{serie} | {letter} | {region}" if (serie or letter or region) else raw,
        "left": serie,
        "letter": letter,
        "right": region,
        "valid": valid,
        "letter_conf": letter_conf,
        "raw_text": raw,
    }

    # save crop if requested
    crop_path = None
    if save_dir is not None:
        try:
            plate_vis = plate_crop.copy()
            # draw bbox around plate in full frame if available
            if bbox is not None:
                # draw on a copy of full frame for context
                full_vis = frame.copy()
                cv2.rectangle(full_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop_path = save_crop(full_vis, save_dir, prefix="plate_full")
            crop_path = save_crop(plate_vis, save_dir, prefix="plate_crop")
            result["crop_path"] = str(crop_path)
        except Exception as e:
            print(f"[WARN] failed saving crop: {e}")

    # post to backend if requested
    if post_url:
        payload = {"timestamp": result["timestamp"], "matricule": result["matricule"], "left": serie, "letter": letter, "right": region}
        try:
            if requests is None:
                print("[WARN] requests not installed; cannot POST to backend")
            else:
                r = requests.post(post_url, json=payload, timeout=5)
                result["post_status"] = r.status_code
        except Exception as e:
            result["post_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Live camera ANPR capture and display")
    parser.add_argument("--source", type=str, default="0", help="Camera source (device index or RTSP URL). Default 0")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between processed frames (sampling rate)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--post-url", type=str, default=os.getenv("ANPR_BACKEND_URL"), help="Optional backend URL to POST results")
    parser.add_argument("--save-dir", type=str, default="data/pipeline_output/live", help="Folder to save plate crops and annotated frames")
    parser.add_argument("--cooldown", type=float, default=6.0, help="Seconds to wait before reporting the same matricule again")
    parser.add_argument("--letter-threshold", type=float, default=0.75, help="Arabic CNN acceptance threshold (0..1)")
    args = parser.parse_args()

    # prepare reader
    reader = EasyOCRReader(letter_model_threshold=args.letter_threshold)

    # open video source
    src = args.source
    try:
        src_int = int(src)
        capture = cv2.VideoCapture(src_int)
    except Exception:
        capture = cv2.VideoCapture(src)

    if not capture.isOpened():
        print(f"ERROR: cannot open video source {src}")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    last_seen = {}

    print("Starting live ANPR. Press Ctrl-C to stop.")
    print("Source:", src)
    print("Backend post URL:", args.post_url)
    print("Save dir:", save_dir)

    try:
        while True:
            t0 = time.time()
            ret, frame = capture.read()
            if not ret or frame is None:
                time.sleep(max(0.1, args.interval))
                continue

            res = process_frame(frame, reader, model_path="models/plate_detector.pt", conf_threshold=args.conf, save_dir=save_dir, post_url=args.post_url)
            if res is None:
                # nothing produced
                print(f"[{iso_ts()}] No result")
            else:
                mat = res.get("matricule", "")
                now = time.time()
                cooldown = args.cooldown
                seen = last_seen.get(mat)
                if mat and (seen is None or (now - seen) >= cooldown):
                    last_seen[mat] = now
                    # clear terminal-friendly one-line output
                    valid_flag = "VALID" if res.get("valid") else "INVALID"
                    left = res.get("left", "")
                    letter = res.get("letter", "")
                    right = res.get("right", "")
                    conf = res.get("letter_conf", 0.0)
                    ts = res.get("timestamp")
                    print(f"[{ts}] {left} | {letter} | {right}    {valid_flag}    letter_conf={conf:.3f}    raw='{res.get('raw_text','')}'")
                    if res.get("crop_path"):
                        print(f"    crop_saved: {res['crop_path']}")
                # else skip duplicate reporting

            # sleep to control processing rate
            dt = time.time() - t0
            to_sleep = max(0.0, args.interval - dt)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("Stopping live ANPR")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
