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
 - Ensure every successful detection is sent to backend (retries + local queue)
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
import threading

import cv2

from anpr_maroc.detection.plate_detector import detect_plate
from anpr_maroc.processing.segmenter import segment_plate_by_layout
from anpr_maroc.ocr.easyocr_reader import EasyOCRReader

try:
    import requests
except Exception:
    requests = None


PENDING_FILE = Path("data/pipeline_output/pending_posts.jsonl")
PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)


def iso_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def save_crop(img, out_dir: Path, prefix: str = "plate") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    p = out_dir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(p), img)
    return p


def _append_pending(payload: dict) -> None:
    try:
        with open(PENDING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] failed to append pending post: {e}")


def _pop_pending_first() -> dict | None:
    # Read first line and rewrite file without it. Return parsed JSON or None.
    try:
        if not PENDING_FILE.exists():
            return None
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        first = lines[0].strip()
        rest = lines[1:]
        # write rest to tmp then atomically replace
        tmp = PENDING_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(rest)
        tmp.replace(PENDING_FILE)
        return json.loads(first)
    except Exception as e:
        print(f"[ERROR] pop_pending failed: {e}")
        return None


def send_post_with_retries(url: str, payload: dict, headers: dict | None = None, max_attempts: int = 5) -> tuple[bool, str]:
    if requests is None:
        return False, "requests-missing"
    attempt = 0
    backoff = 1.0
    while attempt < max_attempts:
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=8)
            if 200 <= r.status_code < 300:
                return True, str(r.status_code)
            else:
                msg = f"status:{r.status_code}"
                print(f"[WARN] post returned {r.status_code}: {r.text}")
        except Exception as e:
            msg = str(e)
            print(f"[WARN] post attempt {attempt+1} failed: {e}")
        attempt += 1
        time.sleep(backoff)
        backoff = min(30.0, backoff * 2)
    return False, msg


class PostWorker(threading.Thread):
    """Background worker that ensures queued posts are delivered.
    It pops entries from PENDING_FILE and attempts to deliver them.
    If delivery fails, it re-queues the failed payload at the end.
    """

    def __init__(self, url: str, headers: dict | None = None, poll_interval: float = 2.0):
        super().__init__(daemon=True)
        self.url = url
        self.headers = headers or {}
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def run(self):
        print(f"PostWorker started, delivering to {self.url}")
        while not self._stop.is_set():
            payload = _pop_pending_first()
            if payload is None:
                time.sleep(self.poll_interval)
                continue
            ok, info = send_post_with_retries(self.url, payload, headers=self.headers, max_attempts=4)
            if not ok:
                # Re-append payload to end of queue for retry later
                print(f"[WARN] failed to deliver payload, requeueing: {info}")
                _append_pending(payload)
                # wait before next try to avoid tight loop
                time.sleep(5.0)
            else:
                print(f"[INFO] delivered payload: {payload.get('matricule')} -> {info}")

    def stop(self):
        self._stop.set()


def process_frame(
    frame,
    reader: EasyOCRReader,
    model_path: str = "models/plate_detector.pt",
    conf_threshold: float = 0.25,
    save_dir: Path | None = None,
    post_url: str | None = None,
    post_headers: dict | None = None,
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

    # enqueue to pending queue if requested
    if post_url:
        payload = {"timestamp": result["timestamp"], "matricule": result["matricule"], "left": serie, "letter": letter, "right": region}
        # attach extra info if available
        if result.get("crop_path"):
            payload["crop_path"] = result["crop_path"]
        # append to persistent queue; worker thread will deliver reliably
        _append_pending({"payload": payload, "meta": {"detected_at": result["timestamp"]}})

    return result


def main():
    parser = argparse.ArgumentParser(description="Live camera ANPR capture and display")
    parser.add_argument("--source", type=str, default="0", help="Camera source (device index or RTSP URL). Default 0")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between processed frames (sampling rate)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--post-url", type=str, default=os.getenv("ANPR_BACKEND_URL"), help="Optional backend URL to POST results")
    parser.add_argument("--auth-token", type=str, default=os.getenv("ANPR_AUTH_TOKEN"), help="Optional bearer token for backend auth")
    parser.add_argument("--save-dir", type=str, default="data/pipeline_output/live", help="Folder to save plate crops and annotated frames")
    parser.add_argument("--cooldown", type=float, default=6.0, help="Seconds to wait before reporting the same matricule again")
    parser.add_argument("--letter-threshold", type=float, default=0.75, help="Arabic CNN acceptance threshold (0..1)")
    args = parser.parse_args()

    # prepare reader
    reader = EasyOCRReader(letter_model_threshold=args.letter_threshold)

    # prepare post worker if requested
    post_worker = None
    post_headers = None
    if args.post_url:
        if requests is None:
            print("[WARN] requests library not available; post delivery disabled")
        else:
            post_headers = {"Content-Type": "application/json"}
            if args.auth_token:
                post_headers["Authorization"] = f"Bearer {args.auth_token}"
            post_worker = PostWorker(args.post_url, headers=post_headers)
            post_worker.start()

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

            res = process_frame(frame, reader, model_path="models/plate_detector.pt", conf_threshold=args.conf, save_dir=save_dir, post_url=args.post_url, post_headers=post_headers)
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
        if post_worker:
            post_worker.stop()


if __name__ == "__main__":
    main()
