"""Isolated RTSP connectivity test — no OCR/YOLO pipeline involved.

Usage:
  python -m anpr_maroc.scripts.test_rtsp_connection

The stream URL comes from ANPR_RTSP_URL (.env), never from this file:
it carries the camera credentials. See .env.example.

Tries the raw URL first (a password may contain a literal '@'); if OpenCV
fails to open it, retries with the '@' in the password percent-encoded
as %40, since some FFmpeg URL parsers choke on an unescaped '@' inside
the userinfo part of an rtsp:// URL.
"""
from __future__ import annotations

import os
import sys
import time
from urllib.parse import quote, urlsplit, urlunsplit

import cv2

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

TEST_DURATION_S = 30
REPORT_INTERVAL_S = 1.5


def percent_encode_userinfo(url: str) -> str:
    """Return `url` with '@' and ':' escaped inside the password.

    Splitting on the LAST '@' is what makes this work when the password
    itself contains one: everything before it is userinfo, the rest is the
    host. urlsplit alone cannot disambiguate that case.
    """
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    userinfo, _, hostport = parts.netloc.rpartition("@")
    user, sep, password = userinfo.partition(":")
    if not sep:
        return url
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{hostport}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def describe_open_failure(cap: cv2.VideoCapture) -> str:
    """Describe why a capture would not open, without adding a second failure.

    ``getBackendName()`` asserts (``api != 0``) when the capture never bound a
    backend at all — precisely the case here, since we only ask once the open
    has failed. Calling it unguarded turned a clean "camera unreachable" into a
    cv2.error traceback that aborted the run before the percent-encoded URL was
    ever tried; the fallback existed but was unreachable in practice.
    """
    try:
        backend = cap.getBackendName() if cap is not None else "unknown"
    except cv2.error:
        backend = "aucun (le flux n'a jamais ete ouvert)"
    return (f"cannot open stream (backend={backend}); likely causes: host unreachable "
            f"(mauvais reseau/VLAN), timeout, RTSP 401 (auth), 404 (bad path), or "
            f"FFmpeg/OpenCV build missing RTSP support")


def mask_credentials(url: str) -> str:
    """Hide the userinfo part so a terminal log never carries the password."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    _, _, hostport = parts.netloc.rpartition("@")
    return urlunsplit((parts.scheme, f"***:***@{hostport}", parts.path, parts.query, parts.fragment))


def try_connect(url: str, label: str) -> bool:
    print(f"\n--- Trying {label}: {mask_credentials(url)} ---")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"ECHEC: {describe_open_failure(cap)}")
        cap.release()
        return False

    print("Connexion ouverte, lecture des frames...")
    start = time.time()
    last_report = 0.0
    frame_count = 0
    fps_window_start = start
    fps_window_count = 0

    try:
        while time.time() - start < TEST_DURATION_S:
            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                print(f"ECHEC en cours de lecture a t={now - start:.1f}s: grab/retrieve a retourne ret={ret}")
                return False

            frame_count += 1
            fps_window_count += 1

            if now - last_report >= REPORT_INTERVAL_S:
                elapsed_window = now - fps_window_start
                fps = fps_window_count / elapsed_window if elapsed_window > 0 else 0.0
                h, w = frame.shape[:2]
                print(f"Frame OK: n°{frame_count} | {w}x{h} | {fps:.1f} FPS")
                last_report = now
                fps_window_start = now
                fps_window_count = 0
    finally:
        cap.release()

    print(f"SUCCES: {frame_count} frames lues en {TEST_DURATION_S}s via {label}")
    return True


def main() -> None:
    raw_url = os.getenv("ANPR_RTSP_URL")
    if not raw_url:
        print(
            "ANPR_RTSP_URL non definie. Renseignez-la dans .env (voir .env.example).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"cv2 version: {cv2.__version__}")
    print(f"Test RTSP pendant {TEST_DURATION_S}s (rapport toutes les {REPORT_INTERVAL_S}s)")

    if try_connect(raw_url, "URL brute (mot de passe avec '@' litteral)"):
        return

    encoded_url = percent_encode_userinfo(raw_url)
    if encoded_url == raw_url:
        print("\nECHEC: rien a encoder dans l'URL, pas de seconde tentative.")
        return

    print("\nURL brute en echec, tentative avec mot de passe encode (%40)...")
    if try_connect(encoded_url, "URL encodee (%40)"):
        return

    print("\nECHEC FINAL: aucune des deux URLs n'a fonctionne.")


if __name__ == "__main__":
    main()
