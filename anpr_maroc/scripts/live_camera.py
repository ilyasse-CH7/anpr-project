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


def send_post_with_retries(url: str, payload: dict, headers: dict | None = None, files_path: str | None = None, max_attempts: int = 5) -> tuple[bool, str]:
    """Send payload reliably. If files_path is provided, perform multipart upload where the JSON payload is placed in the 'payload' form field and the image is sent as file 'image'."""
    if requests is None:
        return False, "requests-missing"
    attempt = 0
    backoff = 1.0
    last_msg = ""
    while attempt < max_attempts:
        try:
            if files_path:
                # multipart: put payload as a JSON string in a form field and attach file
                with open(files_path, "rb") as fh:
                    files = {"image": (Path(files_path).name, fh, "image/jpeg")}
                    data = {"payload": json.dumps(payload, ensure_ascii=False)}
                    r = requests.post(url, data=data, files=files, headers=headers or {}, timeout=10)
            else:
                r = requests.post(url, json=payload, headers=headers or {}, timeout=8)

            if 200 <= r.status_code < 300:
                return True, str(r.status_code)
            else:
                last_msg = f"status:{r.status_code}"
                print(f"[WARN] post returned {r.status_code}: {r.text}")
        except Exception as e:
            last_msg = str(e)
            print(f"[WARN] post attempt {attempt+1} failed: {e}")
        attempt += 1
        time.sleep(backoff)
        backoff = min(30.0, backoff * 2)
    return False, last_msg


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
            item = _pop_pending_first()
            if item is None:
                time.sleep(self.poll_interval)
                continue
            # item can be either a raw payload dict or wrapper {"payload":..., "meta":...}
            if isinstance(item, dict) and "payload" in item:
                payload = item.get("payload")
            else:
                payload = item

            files_path = None
            if isinstance(payload, dict) and payload.get("crop_path"):
                files_path = payload.get("crop_path")

            ok, info = send_post_with_retries(self.url, payload, headers=self.headers, files_path=files_path, max_attempts=4)
            if not ok:
                # Re-append item to end of queue for retry later
                print(f"[WARN] failed to deliver payload, requeueing: {info}")
                _append_pending(item)
                # wait before next try to avoid tight loop
                time.sleep(5.0)
            else:
                # best-effort extract matricule for logging
                mat = None
                try:
                    if isinstance(payload, dict):
                        mat = payload.get("matricule") or payload.get("left")
                except Exception:
                    mat = None
                print(f"[INFO] delivered payload: {mat} -> {info}")

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
    display: bool = True,
):
    # 1. detect plate
    bbox = None
    try:
        bbox = detect_plate(frame, model_path=model_path, conf_threshold=conf_threshold)
    except Exception as e:
        # detection error; treat as no bbox
        print(f"[WARN] detect_plate error: {e}")
        bbox = None

    vis_frame = frame.copy()

    is_already_cropped = False
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        plate_crop = frame[y1:y2, x1:x2]
        # Draw green bounding box for visualization
        if display:
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_frame, "Plaque detectee", (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        plate_crop = frame
        is_already_cropped = True
        if display:
            cv2.putText(vis_frame, "Pas de plaque detectable", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 2. run OCR + segmentation
    try:
        ocr_result = reader.read_and_parse(plate_crop, try_segment=lambda img: segment_plate_by_layout(img))
    except Exception as e:
        print(f"[ERROR] read_and_parse failed: {e}")
        return None, vis_frame if display else None

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

    # Add OCR result text to visualization
    if display and bbox is not None:
        mat_text = result["matricule"]
        y_offset = y2 + 30
        cv2.putText(vis_frame, f"Matricule: {mat_text}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Conf: {letter_conf:.2f} | Valid: {valid}", (x1, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

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

    return result, vis_frame if display else None


def main():
    parser = argparse.ArgumentParser(description="Live camera ANPR capture and display")
    parser.add_argument("--source", type=str, default="0", help="Camera source (device index or RTSP URL). Default 0")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between processed frames (sampling rate)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--post-url", type=str, default=os.getenv("ANPR_BACKEND_URL"), help="Optional backend URL to POST results")
    parser.add_argument("--auth-token", type=str, default=os.getenv("ANPR_AUTH_TOKEN"), help="Optional bearer token for backend auth")
    parser.add_argument("--save-dir", type=str, default="data/pipeline_output/live", help="Folder to save plate crops and annotated frames")
    parser.add_argument("--cooldown", type=float, default=6.0, help="Seconds to wait before reporting the same matricule again; use 0.0 to accept every frame")
    parser.add_argument("--letter-threshold", type=float, default=0.75, help="Arabic CNN acceptance threshold (0..1)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Optional folder to save left/letter/right segmentation debug crops")
    parser.add_argument("--no-display", action="store_true", help="Disable video window display")
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
    display = not args.no_display
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("Starting live ANPR. Press Ctrl-C to stop.")
    print("Source:", src)
    print("Backend post URL:", args.post_url)
    print("Save dir:", save_dir)
    print("Display: ON" if display else "Display: OFF")
    if debug_dir is not None:
        print("Debug dir:", debug_dir)

    try:
        while True:
            t0 = time.time()
            ret, frame = capture.read()
            if not ret or frame is None:
                time.sleep(max(0.1, args.interval))
                continue

            res, vis_frame = process_frame(frame, reader, model_path="models/plate_detector.pt", conf_threshold=args.conf, save_dir=save_dir, post_url=args.post_url, post_headers=post_headers, display=display)
            # Live debug crops for center/left/right segmentation when requested
            if debug_dir is not None:
                try:
                    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
                    debug_sub = debug_dir / ts
                    debug_sub.mkdir(parents=True, exist_ok=True)
                    seg_fn = lambda img: segment_plate_by_layout(img, debug_dir=str(debug_sub))
                    # Re-run segmentation only for debug artifact generation if a plate was detected and has a crop
                    if res is not None and res.get("crop_path"):
                        plate_for_debug = frame.copy()
                        # extract bbox if possible before processing again to keep same crop
                        try:
                            det = detect_plate(plate_for_debug, model_path="models/plate_detector.pt", conf_threshold=args.conf)
                            if det is not None:
                                x1, y1, x2, y2 = det
                                plate_for_debug = plate_for_debug[y1:y2, x1:x2]
                            seg_fn(plate_for_debug)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARN] debug crop generation failed: {e}")
            if res is None:
                # nothing produced
                if display and vis_frame is not None:
                    try:
                        cv2.imshow("ANPR Live", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    except Exception as e:
                        print(f"[WARN] Display error: {e}")
                        display = False
            else:
                mat = res.get("matricule", "")
                now = time.time()
                cooldown = args.cooldown
                seen = last_seen.get(mat)
                if mat and (seen is None or (now - seen) >= cooldown):
                    last_seen[mat] = now
                    # Terminal output: just the matricule
                    print(f"{mat}")
                # else skip duplicate reporting

                # Display video frame with annotations
                if display and vis_frame is not None:
                    try:
                        cv2.imshow("ANPR Live", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    except Exception as e:
                        print(f"[WARN] Display error: {e}")
                        display = False

            # sleep to control processing rate
            dt = time.time() - t0
            to_sleep = max(0.0, args.interval - dt)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("\nStopping live ANPR")
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
        if post_worker:
            post_worker.stop()


if __name__ == "__main__":
    main()
