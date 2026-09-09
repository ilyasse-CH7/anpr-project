"""Harvest Arabic-letter crops from a live RTSP camera to build a training set.

The classifier is undertrained because ``data/arabic_letters/real`` holds a
handful of samples.  This script fills ``data/arabic_letters/to_label`` with
unlabelled letter crops so the existing annotator can label them in bulk:

    python -m anpr_maroc.scripts.harvest_arabic_letters --ip 192.168.1.64
    python -m anpr_maroc.scripts.harvest_arabic_letters --source rtsp://... --max-crops 500
    python data/arabic_letters/annotator/serve_annotator.py   # then label them

Design notes for an unattended run:
 - A grabber thread keeps only the newest frame.  Reading an RTSP socket slower
   than it produces frames otherwise builds an ever-growing latency backlog.
 - The stream is read over TCP; UDP packet loss produces smeared frames that
   would silently poison the dataset.
 - Crops are quality-gated (size, sharpness, glyph validity) and deduplicated,
   because a stopped car yields hundreds of identical crops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import typing as t
import urllib.parse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anpr_maroc import config as _config
from anpr_maroc.detection.plate_detector import PlateDetector
from anpr_maroc.ocr.arabic_letter_model import (
    ArabicLetterClassifier,
    DEFAULT_MODEL_PATH,
    preprocess_letter,
)
from anpr_maroc.processing.segmenter import detect_plate_layout, segment_plate_by_layout

DEFAULT_OUTPUT = Path("data/arabic_letters/to_label")
DEFAULT_CONTEXT = Path("data/arabic_letters/context")
METADATA_NAME = "harvest_metadata.jsonl"


def build_rtsp_url(args: argparse.Namespace) -> str:
    """Resolve the stream URL from an explicit source, .env, or Hikvision parts."""
    if args.source:
        return args.source
    if _config.CAMERA_RTSP_URL:
        return _config.CAMERA_RTSP_URL

    ip = args.ip or _config.CAMERA_IP
    if not ip:
        raise SystemExit(
            "No stream source. Pass --source/--ip, or set CAMERA_RTSP_URL or "
            "CAMERA_IP (with RTSP_USER/RTSP_PASS) in your .env."
        )
    user = args.user or _config.RTSP_USER
    password = args.password or _config.RTSP_PASS
    # Credentials routinely contain '@' or '/', which would truncate the URL.
    credentials = ""
    if user:
        credentials = urllib.parse.quote(user, safe="")
        if password:
            credentials += ":" + urllib.parse.quote(password, safe="")
        credentials += "@"
    return f"rtsp://{credentials}{ip}:{args.port}/Streaming/Channels/{args.channel}"


def mask_url(url: str) -> str:
    """Hide the password so a long-running console log stays shareable."""
    return urllib.parse.urlsplit(url)._replace(
        netloc=urllib.parse.urlsplit(url).netloc.split("@")[-1]
    ).geturl() if "@" in url else url


class FrameGrabber(threading.Thread):
    """Continuously read a video source, exposing only the most recent frame.

    Reconnects with exponential backoff so an unattended harvest survives the
    network blips and camera reboots expected over a multi-hour session.
    """

    def __init__(self, source: str, reconnect_delay: float = 2.0, max_delay: float = 30.0):
        super().__init__(daemon=True)
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()
        self._frame: t.Optional[np.ndarray] = None
        self._seq = 0
        self._stop = threading.Event()
        self.connected = threading.Event()

    def _open(self) -> t.Optional[cv2.VideoCapture]:
        try:
            source: t.Union[int, str] = int(self.source)
        except (TypeError, ValueError):
            source = self.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return None
        # Keep the driver-side queue minimal; we do our own frame dropping.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def run(self) -> None:
        delay = self.reconnect_delay
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self.connected.clear()
                print(f"[WARN] cannot open stream, retrying in {delay:.0f}s")
                self._stop.wait(delay)
                delay = min(self.max_delay, delay * 2)
                continue

            print("[INFO] stream connected")
            self.connected.set()
            delay = self.reconnect_delay
            failures = 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    # Tolerate isolated dropped packets, but treat a sustained
                    # run of failures as a dead connection.
                    if failures > 60:
                        print("[WARN] stream stalled, reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            capture.release()
            self.connected.clear()

    def latest(self) -> t.Tuple[int, t.Optional[np.ndarray]]:
        with self._lock:
            return self._seq, None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()


def dhash(image: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash, robust to the exposure jitter between frames."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian; low values mean motion blur or defocus."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest Arabic-letter crops from an RTSP camera into the annotation queue."
    )
    source = parser.add_argument_group("stream")
    source.add_argument("--source", type=str, default=None, help="Full stream URL, file path, or device index. Overrides --ip.")
    source.add_argument("--ip", type=str, default=None, help="Hikvision camera IP (uses CAMERA_IP otherwise).")
    source.add_argument("--user", type=str, default=None, help="RTSP user (uses RTSP_USER otherwise).")
    source.add_argument("--password", type=str, default=None, help="RTSP password (uses RTSP_PASS otherwise).")
    source.add_argument("--port", type=int, default=554, help="RTSP port.")
    source.add_argument("--channel", type=str, default="101", help="Hikvision channel: 101 = main stream, 102 = sub-stream.")
    source.add_argument("--udp", action="store_true", help="Use UDP transport instead of TCP (lossier, not advised for datasets).")

    quality = parser.add_argument_group("quality gates")
    quality.add_argument("--conf", type=float, default=0.25, help="YOLO plate detection confidence threshold.")
    quality.add_argument("--min-size", type=int, default=24, help="Reject letter crops smaller than this, in pixels.")
    quality.add_argument("--min-sharpness", type=float, default=100.0, help="Reject blurry crops below this Laplacian variance. Scale- and camera-dependent: calibrate from the sharpness field in the metadata after a short run.")
    quality.add_argument("--dedup-distance", type=int, default=8, help="Reject a crop within this Hamming distance of a recent one.")
    quality.add_argument("--dedup-window", type=int, default=800, help="Number of recent hashes compared against.")

    run = parser.add_argument_group("run control")
    run.add_argument("--interval", type=float, default=0.4, help="Seconds between processed frames.")
    run.add_argument("--max-crops", type=int, default=0, help="Stop after saving this many crops (0 = unlimited).")
    run.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds (0 = unlimited).")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where letter crops are written.")
    run.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT, help="Where the matching full-plate crops are written.")
    run.add_argument("--no-context", action="store_true", help="Do not save the full-plate context image.")
    run.add_argument("--model", type=str, default="models/plate_detector.pt", help="YOLO weights.")
    run.add_argument("--letter-model", type=str, default=str(DEFAULT_MODEL_PATH), help="CNN used only to record a guess in metadata.")
    run.add_argument("--display", action="store_true", help="Show an annotated preview window.")
    run.add_argument("--stats-interval", type=float, default=30.0, help="Seconds between statistics lines.")
    args = parser.parse_args()

    # Must precede any VideoCapture construction: OpenCV reads it at open time.
    if not args.udp:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    url = build_rtsp_url(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_context:
        args.context_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.parent / METADATA_NAME

    # skip_if_cropped would make a wide camera frame look like an already-cropped
    # plate and suppress detection entirely, so it is disabled here.
    detector = PlateDetector(
        model_path=args.model, conf_threshold=args.conf, skip_if_cropped=False
    )
    classifier = ArabicLetterClassifier(args.letter_model)

    grabber = FrameGrabber(url)
    grabber.start()

    hashes: deque[int] = deque(maxlen=args.dedup_window)
    counters = {"frames": 0, "no_plate": 0, "no_segment": 0, "too_small": 0, "blurry": 0, "no_glyph": 0, "duplicate": 0, "saved": 0}
    started = time.time()
    last_stats = started
    last_seq = -1

    print(f"[INFO] source      : {mask_url(url)}")
    print(f"[INFO] crops       -> {args.output}")
    if not args.no_context:
        print(f"[INFO] context     -> {args.context_dir}")
    print(f"[INFO] metadata    -> {metadata_path}")
    print(f"[INFO] letter CNN  : {'loaded' if classifier.available else 'unavailable (guess omitted)'}")
    print("[INFO] harvesting, Ctrl-C to stop")

    def print_stats() -> None:
        elapsed = max(1e-6, time.time() - started)
        rate = counters["saved"] / (elapsed / 60.0)
        print(
            f"[STATS] {counters['saved']} saved ({rate:.1f}/min) | frames={counters['frames']} "
            f"no_plate={counters['no_plate']} no_seg={counters['no_segment']} small={counters['too_small']} "
            f"blur={counters['blurry']} no_glyph={counters['no_glyph']} dup={counters['duplicate']}"
        )

    metadata_file = metadata_path.open("a", encoding="utf-8")
    try:
        while True:
            loop_start = time.time()
            if args.duration and (loop_start - started) >= args.duration:
                print("[INFO] duration reached")
                break

            seq, frame = grabber.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.05)
                continue
            last_seq = seq
            counters["frames"] += 1

            preview = frame if args.display else None
            try:
                bbox = detector.detect(frame)
            except Exception as exc:
                print(f"[WARN] detection failed: {exc}")
                bbox = None

            letter_crop = None
            plate_crop = None
            if bbox is None:
                counters["no_plate"] += 1
            else:
                x1, y1, x2, y2 = bbox
                plate_crop = frame[y1:y2, x1:x2]
                if args.display:
                    preview = frame.copy()
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                try:
                    zones = segment_plate_by_layout(plate_crop)
                except Exception:
                    zones = []
                if len(zones) != 3 or zones[1] is None or zones[1].size == 0:
                    counters["no_segment"] += 1
                else:
                    letter_crop = zones[1]

            if letter_crop is not None:
                height, width = letter_crop.shape[:2]
                if min(height, width) < args.min_size:
                    counters["too_small"] += 1
                    letter_crop = None

            if letter_crop is not None:
                sharp = sharpness(letter_crop)
                if sharp < args.min_sharpness:
                    counters["blurry"] += 1
                    letter_crop = None

            if letter_crop is not None:
                # The CNN's own preprocessing is the strictest validity test we
                # have: it fails when the crop holds no isolated glyph, which
                # catches empty zones and mis-segmented plate frames.
                try:
                    preprocess_letter(letter_crop)
                except (ValueError, cv2.error):
                    counters["no_glyph"] += 1
                    letter_crop = None

            if letter_crop is not None:
                digest = dhash(letter_crop)
                if any(hamming(digest, seen) <= args.dedup_distance for seen in hashes):
                    counters["duplicate"] += 1
                    letter_crop = None

            if letter_crop is not None:
                hashes.append(digest)
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                stem = f"plate_{timestamp}_{digest & 0xFFFFFF:06x}"
                crop_path = args.output / f"{stem}.png"
                if not cv2.imwrite(str(crop_path), letter_crop):
                    raise OSError(f"could not write {crop_path}")

                context_path = None
                if not args.no_context and plate_crop is not None:
                    # Kept so a future segmenter can re-extract this sample
                    # without another harvesting session.
                    context_path = args.context_dir / f"{stem}_plate.png"
                    cv2.imwrite(str(context_path), plate_crop)

                guess, guess_conf = ("", 0.0)
                if classifier.available:
                    guess, guess_conf = classifier.predict(letter_crop)

                try:
                    layout = detect_plate_layout(plate_crop)
                except Exception:
                    layout = "unknown"

                metadata_file.write(json.dumps({
                    "file": crop_path.name,
                    "context": context_path.name if context_path else None,
                    "captured_at": datetime.now().isoformat(timespec="seconds"),
                    "plate_bbox": list(bbox) if bbox else None,
                    "layout": layout,
                    "size": [int(letter_crop.shape[1]), int(letter_crop.shape[0])],
                    "sharpness": round(sharp, 1),
                    "dhash": f"{digest:016x}",
                    # Recorded only for later analysis; files are never sorted
                    # by it, so a wrong guess cannot bias the labelling.
                    "model_guess": guess,
                    "model_conf": round(guess_conf, 3),
                }, ensure_ascii=False) + "\n")
                metadata_file.flush()

                counters["saved"] += 1
                print(f"[SAVE] {crop_path.name}  sharp={sharp:.0f}  guess={guess or '-'} ({guess_conf:.2f})")

                if args.max_crops and counters["saved"] >= args.max_crops:
                    print("[INFO] crop target reached")
                    break

            if args.display and preview is not None:
                cv2.imshow("harvest (q to quit)", preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if (time.time() - last_stats) >= args.stats_interval:
                print_stats()
                last_stats = time.time()

            time.sleep(max(0.0, args.interval - (time.time() - loop_start)))
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        grabber.stop()
        metadata_file.close()
        if args.display:
            cv2.destroyAllWindows()
        print_stats()
        print(f"[INFO] label them with: python {Path('data/arabic_letters/annotator/serve_annotator.py')}")


if __name__ == "__main__":
    main()
