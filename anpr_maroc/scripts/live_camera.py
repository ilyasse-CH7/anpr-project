"""Capture live ANPR : RTSP → YOLO → segmentation → PaddleOCR + CNN lettre.

Le terminal n'affiche QUE les matricules confirmés :

    [14:32:07] Plaque détectée : 13456-ب-27

Une ligne n'est émise qu'après :
  1. consensus multi-frames par champ (left / letter / right) sur une fenêtre
     glissante par plaque suivie,
  2. validation regex stricte du format marocain (processing/validator.py).

Tout le reste (traces YOLO/OpenCV/Paddle, lectures brutes, rejets) est masqué
par défaut et ne réapparaît qu'avec --verbose.

Exemples :
  source .venv/bin/activate && export PYTHONPATH=$PWD
  python -m anpr_maroc.scripts.live_camera --no-display
  python -m anpr_maroc.scripts.live_camera --source 0 --verbose
  python -m anpr_maroc.scripts.live_camera --no-display --duration 60 --stats
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
import typing as t
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

# Ces variables doivent être posées AVANT l'import de paddle/ultralytics :
# elles coupent les bannières d'initialisation écrites directement sur la
# sortie standard, que le terminal propre (phase D) ne doit pas contenir.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_call_stack_level", "0")
os.environ.setdefault("YOLO_VERBOSE", "False")

# L'URL RTSP (identifiants caméra) vit dans .env, jamais dans le dépôt.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv est optionnel : --source reste utilisable.
    pass

import cv2

from anpr_maroc.detection.plate_detector import detect_plate
from anpr_maroc.processing.segmenter import segment_plate_by_layout
from anpr_maroc.processing.validator import ValidationResult, validate_fields

try:
    import requests
except Exception:
    requests = None


PENDING_FILE = Path("data/pipeline_output/pending_posts.jsonl")

# --- Phase A : seuil CNN lettre -------------------------------------------
#
# Le seuil historique de 0.75 rejetait systématiquement des lectures correctes
# sur les plaques de test imprimées basse résolution (score CNN observé
# ~0.60-0.62 alors que la lettre était juste). Le défaut est donc abaissé à
# 0.55.
#
# Garde-fou : ce niveau de confiance N'EST JAMAIS suffisant sur une seule
# frame. Il n'est autorisé que combiné au vote multi-frames (phase C) — au
# moins MIN_VOTES_FOR_LOW_THRESHOLD frames concordantes sur la même plaque
# suivie. Si l'utilisateur désactive ou réduit le vote sous ce plancher, le
# seuil est automatiquement remonté à SAFE_LETTER_THRESHOLD : on n'accepte pas
# une lecture 0.55 non confirmée.
LOW_LETTER_THRESHOLD = 0.55
SAFE_LETTER_THRESHOLD = 0.75
MIN_VOTES_FOR_LOW_THRESHOLD = 3

VERBOSE = False


def log(message: str) -> None:
    """Trace de debug : visible uniquement en --verbose (phase D)."""
    if VERBOSE:
        print(message, file=sys.stderr)


def iso_ts() -> str:
    return datetime.now().astimezone().isoformat()


def _silence_third_party_logs() -> None:
    """Coupe les loggers bavards des dépendances (phase D)."""
    for name in ("ppocr", "paddle", "paddlex", "ultralytics", "PIL", "urllib3", "httpx"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass


@contextlib.contextmanager
def _suppressed_output():
    """Coupe stdout/stderr au niveau descripteur de fichier.

    PaddleOCR écrit ses bannières d'initialisation depuis du code C++ : une
    simple ``contextlib.redirect_stdout`` (qui n'agit que sur ``sys.stdout``)
    ne les intercepte pas. On détourne donc les fd 1 et 2 vers /dev/null le
    temps du chargement des modèles.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def save_crop(img, out_dir: Path, prefix: str = "plate") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = out_dir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), img)
    return path


# --------------------------------------------------------------------------
# File d'envoi backend (inchangée : livraison fiable des matricules validés)
# --------------------------------------------------------------------------


def _append_pending(payload: dict) -> None:
    try:
        PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[ERROR] failed to append pending post: {e}")


def _pop_pending_first() -> dict | None:
    try:
        if not PENDING_FILE.exists():
            return None
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        first, rest = lines[0].strip(), lines[1:]
        tmp = PENDING_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(rest)
        tmp.replace(PENDING_FILE)
        return json.loads(first)
    except Exception as e:
        log(f"[ERROR] pop_pending failed: {e}")
        return None


def send_post_with_retries(
    url: str,
    payload: dict,
    headers: dict | None = None,
    files_path: str | None = None,
    max_attempts: int = 5,
) -> tuple[bool, str]:
    """Envoi fiable ; si files_path est fourni, upload multipart (JSON dans le
    champ 'payload', image dans le champ 'image')."""
    if requests is None:
        return False, "requests-missing"
    attempt = 0
    backoff = 1.0
    last_msg = ""
    while attempt < max_attempts:
        try:
            if files_path:
                with open(files_path, "rb") as fh:
                    files = {"image": (Path(files_path).name, fh, "image/jpeg")}
                    data = {"payload": json.dumps(payload, ensure_ascii=False)}
                    r = requests.post(url, data=data, files=files, headers=headers or {}, timeout=10)
            else:
                r = requests.post(url, json=payload, headers=headers or {}, timeout=8)
            if 200 <= r.status_code < 300:
                return True, str(r.status_code)
            last_msg = f"status:{r.status_code}"
            log(f"[WARN] post returned {r.status_code}: {r.text}")
        except Exception as e:
            last_msg = str(e)
            log(f"[WARN] post attempt {attempt + 1} failed: {e}")
        attempt += 1
        time.sleep(backoff)
        backoff = min(30.0, backoff * 2)
    return False, last_msg


class PostWorker(threading.Thread):
    """Livre en tâche de fond les payloads accumulés dans PENDING_FILE."""

    def __init__(self, url: str, headers: dict | None = None, poll_interval: float = 2.0):
        super().__init__(daemon=True)
        self.url = url
        self.headers = headers or {}
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def run(self):
        log(f"[INFO] PostWorker started -> {self.url}")
        while not self._stop.is_set():
            item = _pop_pending_first()
            if item is None:
                time.sleep(self.poll_interval)
                continue
            payload = item.get("payload") if isinstance(item, dict) and "payload" in item else item
            files_path = payload.get("crop_path") if isinstance(payload, dict) else None
            ok, info = send_post_with_retries(
                self.url, payload, headers=self.headers, files_path=files_path, max_attempts=4
            )
            if not ok:
                log(f"[WARN] delivery failed, requeueing: {info}")
                _append_pending(item)
                time.sleep(5.0)
            else:
                log(f"[INFO] delivered: {payload.get('matricule') if isinstance(payload, dict) else '?'} -> {info}")

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Phase C : consensus multi-frames
# --------------------------------------------------------------------------


class PlateTrack:
    """Fenêtre glissante des lectures d'une même plaque suivie."""

    def __init__(self, track_id: int, bbox: tuple[int, int, int, int], window: int):
        self.id = track_id
        self.bbox = bbox
        self.readings: deque[dict] = deque(maxlen=window)
        self.last_update = time.time()
        self.last_emitted = ""
        self.last_emit_time = 0.0

    def add(self, reading: dict, bbox: tuple[int, int, int, int]) -> None:
        self.readings.append(reading)
        self.bbox = bbox
        self.last_update = time.time()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateVoter:
    """Associe chaque détection à une plaque suivie et décide du consensus.

    Le suivi est volontairement simple (recouvrement de boîtes entre frames
    consécutives) : à la cadence d'échantillonnage utilisée, une plaque bouge
    peu d'une frame traitée à la suivante, et il n'y a en pratique qu'un
    véhicule à la fois dans le champ.
    """

    def __init__(
        self,
        window: int = 8,
        min_votes: int = 3,
        iou_threshold: float = 0.3,
        track_ttl: float = 4.0,
        repeat_cooldown: float = 6.0,
    ):
        self.window = window
        self.min_votes = min_votes
        self.iou_threshold = iou_threshold
        self.track_ttl = track_ttl
        self.repeat_cooldown = repeat_cooldown
        self.tracks: list[PlateTrack] = []
        self._next_id = 1

    def _expire(self) -> None:
        now = time.time()
        self.tracks = [tr for tr in self.tracks if (now - tr.last_update) <= self.track_ttl]

    def _match(self, bbox: tuple[int, int, int, int]) -> PlateTrack:
        best, best_iou = None, 0.0
        for tr in self.tracks:
            score = _iou(tr.bbox, bbox)
            if score > best_iou:
                best, best_iou = tr, score
        if best is not None and best_iou >= self.iou_threshold:
            return best
        track = PlateTrack(self._next_id, bbox, self.window)
        self._next_id += 1
        self.tracks.append(track)
        return track

    @staticmethod
    def _majority(values: list[str]) -> tuple[str, int]:
        """Valeur majoritaire non vide et son nombre de voix."""
        counts = Counter(v for v in values if v)
        if not counts:
            return "", 0
        value, votes = counts.most_common(1)[0]
        return value, votes

    def submit(
        self, bbox: tuple[int, int, int, int], reading: dict
    ) -> tuple[PlateTrack, t.Optional[dict]]:
        """Ajoute une lecture et retourne le résultat consolidé s'il est prêt.

        Retourne (track, None) tant que le consensus n'est pas atteint ou que
        le matricule consolidé ne passe pas la validation regex (phase B).
        """
        self._expire()
        track = self._match(bbox)
        track.add(reading, bbox)

        if len(track.readings) < self.min_votes:
            return track, None

        left, left_votes = self._majority([r["left"] for r in track.readings])
        letter, letter_votes = self._majority([r["letter"] for r in track.readings])
        right, right_votes = self._majority([r["right"] for r in track.readings])
        votes = min(left_votes, letter_votes, right_votes)

        # Majorité STRICTE, pas seulement `min_votes` occurrences : sur une
        # plaque limite l'OCR alterne entre deux lectures plausibles (observé
        # en conditions réelles : '34250-أ-8' vs '3425-أ-6'), toutes deux
        # valides au regard de la regex. Exiger plus de la moitié de la
        # fenêtre empêche qu'une variante minoritaire soit confirmée ; en cas
        # d'égalité, aucune n'est émise et on attend d'autres frames.
        required = max(self.min_votes, len(track.readings) // 2 + 1)
        if votes < required:
            log(
                f"[VOTE] track#{track.id} pas de consensus ({votes}/{len(track.readings)}, requis {required}) "
                f"left={left!r}:{left_votes} letter={letter!r}:{letter_votes} right={right!r}:{right_votes}"
            )
            return track, None

        layout, _ = self._majority([r.get("layout", "") for r in track.readings])
        result: ValidationResult = validate_fields(left, letter, right, layout=layout or None)
        # `result.partiel` : chiffres validés mais lettre sous le seuil du CNN.
        # On l'émet quand même, explicitement marquée — taire la lecture
        # priverait l'opérateur d'une information exacte à 2 champs sur 3.
        if not result.valid and not result.partiel:
            log(f"[REJET] track#{track.id} {left}-{letter}-{right} : {result.reason}")
            return track, None

        # Anti-doublon : même matricule déjà émis récemment pour cette plaque.
        now = time.time()
        if track.last_emitted == result.matricule and (now - track.last_emit_time) < self.repeat_cooldown:
            return track, None
        track.last_emitted = result.matricule
        track.last_emit_time = now

        agreeing = [
            r
            for r in track.readings
            if r["left"] == left and r["letter"] == letter and r["right"] == right
        ]
        confidences = [r.get("confidence", 0.0) for r in agreeing] or [0.0]
        letter_confs = [r.get("letter_conf", 0.0) for r in agreeing] or [0.0]

        return track, {
            "matricule": result.matricule,
            "affichage": result.affichage(),
            "left": result.left,
            "letter": result.letter,
            "right": result.right,
            "letter_known": result.letter_known,
            "layout": result.layout,
            "votes": votes,
            "window": len(track.readings),
            "confidence": sum(confidences) / len(confidences),
            "letter_conf": sum(letter_confs) / len(letter_confs),
            "timestamp": iso_ts(),
        }


# --------------------------------------------------------------------------
# Traitement d'une frame
# --------------------------------------------------------------------------


def process_frame(
    frame,
    reader,
    model_path: str = "models/plate_detector.pt",
    conf_threshold: float = 0.25,
    display: bool = True,
) -> tuple[t.Optional[dict], t.Any, dict]:
    """Détecte + lit une plaque sur une frame.

    Retourne (lecture_brute | None, frame annotée | None, timings).
    La lecture brute n'est JAMAIS affichée : elle alimente le vote (phase C).
    """
    timings = {"detect": 0.0, "ocr": 0.0}

    t0 = time.perf_counter()
    try:
        bbox = detect_plate(frame, model_path=model_path, conf_threshold=conf_threshold)
    except Exception as e:
        log(f"[WARN] detect_plate error: {e}")
        bbox = None
    timings["detect"] = time.perf_counter() - t0

    vis_frame = frame.copy() if display else None

    if bbox is None:
        # Pas de plaque localisée : on ne lance surtout pas l'OCR sur la frame
        # entière (plusieurs minutes par frame sur CPU).
        return None, vis_frame, timings

    x1, y1, x2, y2 = bbox
    plate_crop = frame[y1:y2, x1:x2]
    if display:
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    t0 = time.perf_counter()
    try:
        ocr_result = reader.read_and_parse(plate_crop, try_segment=segment_plate_by_layout)
    except Exception as e:
        log(f"[ERROR] read_and_parse failed: {e}")
        timings["ocr"] = time.perf_counter() - t0
        return None, vis_frame, timings
    timings["ocr"] = time.perf_counter() - t0

    parsed = ocr_result.get("parsed", {}) or {}
    zone_conf = {zr.get("zone"): float(zr.get("conf", 0.0)) for zr in (ocr_result.get("zone_results") or [])}
    confs = [c for c in zone_conf.values() if c > 0]

    reading = {
        "left": parsed.get("left", ""),
        "letter": parsed.get("letter", ""),
        "right": parsed.get("right", ""),
        "layout": ocr_result.get("layout", ""),
        "letter_conf": zone_conf.get("letter", 0.0),
        "confidence": (sum(confs) / len(confs)) if confs else 0.0,
        "raw_text": ocr_result.get("raw_text", ""),
        "bbox": bbox,
        "crop": plate_crop,
    }
    log(
        f"[BRUT] left={reading['left']!r} letter={reading['letter']!r} "
        f"right={reading['right']!r} layout={reading['layout']!r} "
        f"conf={reading['confidence']:.2f} letter_conf={reading['letter_conf']:.2f}"
    )
    if display:
        cv2.putText(
            vis_frame,
            f"{reading['left']}-{reading['letter']}-{reading['right']}",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return reading, vis_frame, timings


# --------------------------------------------------------------------------


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Live camera ANPR capture")
    # Aucune URL par défaut : les identifiants de la caméra n'ont pas à vivre
    # dans le dépôt. La source vient de ANPR_RTSP_URL (fichier .env) ou de
    # --source. Voir .env.example.
    default_source = os.getenv("ANPR_RTSP_URL")
    parser.add_argument("--source", type=str, default=default_source, help="Index caméra ou URL RTSP (défaut : $ANPR_RTSP_URL)")
    parser.add_argument("--interval", type=float, default=0.0, help="Secondes entre deux frames traitées (0 = au fil de l'eau)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance YOLO")
    parser.add_argument("--post-url", type=str, default=os.getenv("ANPR_BACKEND_URL"), help="URL backend optionnelle (POST des matricules validés)")
    parser.add_argument("--auth-token", type=str, default=os.getenv("ANPR_AUTH_TOKEN"), help="Bearer token backend optionnel")
    parser.add_argument("--save-dir", type=str, default="data/pipeline_output/live", help="Dossier des crops de plaques validées")
    parser.add_argument("--letter-threshold", type=float, default=LOW_LETTER_THRESHOLD, help=f"Seuil CNN lettre arabe (défaut {LOW_LETTER_THRESHOLD}, autorisé uniquement avec vote multi-frames)")
    parser.add_argument("--vote-window", type=int, default=8, help="Taille de la fenêtre glissante par plaque suivie")
    parser.add_argument("--min-votes", type=int, default=3, help="Nombre de frames concordantes requises par champ")
    parser.add_argument("--cooldown", type=float, default=6.0, help="Secondes avant de ré-annoncer le même matricule")
    parser.add_argument("--no-display", action="store_true", help="Désactive la fenêtre vidéo")
    parser.add_argument("--no-db", action="store_true", help="Désactive l'insertion en base")
    parser.add_argument("--duration", type=float, default=0.0, help="Arrêt automatique après N secondes (0 = illimité)")
    parser.add_argument("--stats", action="store_true", help="Affiche le rapport FPS / temps de traitement à l'arrêt")
    parser.add_argument("--verbose", action="store_true", help="Réactive les traces de debug (lectures brutes, rejets, backend)")
    args = parser.parse_args()

    if not args.source:
        parser.error(
            "aucune source vidéo. Renseignez ANPR_RTSP_URL dans .env "
            "(voir .env.example) ou passez --source <url_rtsp|index_webcam>."
        )

    VERBOSE = args.verbose
    if not VERBOSE:
        _silence_third_party_logs()

    # Phase A : garde-fou. Un seuil bas n'est jamais accepté sans vote.
    letter_threshold = args.letter_threshold
    if letter_threshold < SAFE_LETTER_THRESHOLD and args.min_votes < MIN_VOTES_FOR_LOW_THRESHOLD:
        log(
            f"[WARN] --min-votes={args.min_votes} < {MIN_VOTES_FOR_LOW_THRESHOLD} : "
            f"seuil lettre remonté {letter_threshold} -> {SAFE_LETTER_THRESHOLD}"
        )
        letter_threshold = SAFE_LETTER_THRESHOLD

    # L'initialisation de PaddleOCR écrit sur stdout : on la capture pour
    # garder un terminal strictement propre.
    from anpr_maroc.ocr.paddleocr_reader import PaddleOCRReader

    if VERBOSE:
        reader = PaddleOCRReader(letter_model_threshold=letter_threshold)
    else:
        with _suppressed_output():
            reader = PaddleOCRReader(letter_model_threshold=letter_threshold)

    store = None
    if not args.no_db:
        try:
            from anpr_maroc.backend.storage import get_store

            store = get_store()
            log(f"[INFO] stockage: {store.describe()}")
        except Exception as e:
            print(f"[ERREUR] base de données indisponible : {e}", file=sys.stderr)
            store = None

    post_worker = None
    if args.post_url:
        if requests is None:
            log("[WARN] requests indisponible ; envoi backend désactivé")
        else:
            headers = {"Content-Type": "application/json"}
            if args.auth_token:
                headers["Authorization"] = f"Bearer {args.auth_token}"
            post_worker = PostWorker(args.post_url, headers=headers)
            post_worker.start()

    src = args.source
    try:
        capture = cv2.VideoCapture(int(src))
    except ValueError:
        capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        print(f"[ERREUR] source vidéo inaccessible : {src}", file=sys.stderr)
        sys.exit(1)

    save_dir = Path(args.save_dir)
    display = not args.no_display
    voter = PlateVoter(
        window=args.vote_window,
        min_votes=args.min_votes,
        repeat_cooldown=args.cooldown,
    )

    log(f"[INFO] source={src} seuil_lettre={letter_threshold} fenetre={args.vote_window} min_votes={args.min_votes}")

    # Phase G : instrumentation.
    frame_times: list[float] = []
    detect_times: list[float] = []
    ocr_times: list[float] = []
    frames_with_plate = 0
    grabbed = 0
    started = time.time()

    try:
        while True:
            if args.duration and (time.time() - started) >= args.duration:
                break

            loop_start = time.perf_counter()
            ret, frame = capture.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            grabbed += 1

            reading, vis_frame, timings = process_frame(
                frame, reader, conf_threshold=args.conf, display=display
            )
            frame_times.append(time.perf_counter() - loop_start)
            detect_times.append(timings["detect"])
            if timings["ocr"] > 0:
                ocr_times.append(timings["ocr"])
                frames_with_plate += 1

            if reading is not None:
                _track, confirmed = voter.submit(reading["bbox"], reading)
                if confirmed is not None:
                    crop_path = None
                    try:
                        crop_path = str(save_crop(reading["crop"], save_dir, prefix="plate"))
                    except Exception as e:
                        log(f"[WARN] échec sauvegarde crop: {e}")

                    if store is not None:
                        try:
                            store.insert_plate(
                                matricule=confirmed["matricule"],
                                confiance=confirmed["confidence"],
                                chemin_image_crop=crop_path,
                            )
                        except Exception as e:
                            print(f"[ERREUR] insertion base échouée : {e}", file=sys.stderr)

                    if args.post_url:
                        _append_pending(
                            {
                                "payload": {
                                    "timestamp": confirmed["timestamp"],
                                    "matricule": confirmed["matricule"],
                                    "left": confirmed["left"],
                                    "letter": confirmed["letter"],
                                    "right": confirmed["right"],
                                    "letter_known": confirmed["letter_known"],
                                    "confiance": confirmed["confidence"],
                                    "crop_path": crop_path,
                                }
                            }
                        )

                    # --- SEULE sortie standard du programme (phase D) ---
                    # Une lettre indéterminée est annoncée comme telle : le
                    # libellé ne doit jamais laisser croire à une lecture sûre.
                    etiquette = (
                        "Plaque détectée" if confirmed["letter_known"] else "Plaque partielle"
                    )
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"{etiquette} : {confirmed['affichage']}",
                        flush=True,
                    )
                    log(
                        f"[INFO] votes={confirmed['votes']}/{confirmed['window']} "
                        f"conf={confirmed['confidence']:.2f} lettre={confirmed['letter_conf']:.2f} "
                        f"format={confirmed['layout']} crop={crop_path}"
                    )

            if display and vis_frame is not None:
                try:
                    cv2.imshow("ANPR Live", vis_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except Exception as e:
                    log(f"[WARN] affichage indisponible: {e}")
                    display = False

            if args.interval:
                time.sleep(max(0.0, args.interval - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
        if post_worker:
            post_worker.stop()

        if args.stats or VERBOSE:
            elapsed = time.time() - started
            n = len(frame_times) or 1
            avg_frame = sum(frame_times) / n
            avg_detect = sum(detect_times) / n
            avg_ocr = (sum(ocr_times) / len(ocr_times)) if ocr_times else 0.0
            print(
                "\n--- Mesure de performance ---\n"
                f"Durée du run              : {elapsed:.1f} s\n"
                f"Frames traitées           : {len(frame_times)} (dont {frames_with_plate} avec plaque détectée)\n"
                f"Temps moyen / frame       : {avg_frame * 1000:.1f} ms  ->  {1.0 / avg_frame if avg_frame else 0:.2f} FPS\n"
                f"  dont détection YOLO     : {avg_detect * 1000:.1f} ms\n"
                f"  dont OCR+CNN (si plaque): {avg_ocr * 1000:.1f} ms\n"
                f"Frames lues depuis la source : {grabbed}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
