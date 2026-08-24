import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from ultralytics import YOLO
from collections import Counter
import urllib.request
import re
import requests
import os
import sys
import time
import logging
from dotenv import load_dotenv


load_dotenv()

# Configuration
TEXT_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", 0.40))
PLATE_MIN_VOTES = int(os.getenv("PLATE_MIN_VOTES", 3))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

# IMPORTANT : le niveau de logging doit dépendre de DEBUG_MODE,
# sinon logging.debug(...) ne s'affiche jamais.
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

vehicle_model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]

model_url = "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt"
model_path = "../models/license-plate-finetune-v1n.pt"
if not os.path.exists(model_path):
    os.makedirs("../models", exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as e:
        logging.warning(f"Impossible de télécharger le modèle de plaque: {e}")
plate_model = YOLO(model_path)

PLATE_REGEX = re.compile(r'^[\d\u0600-\u06FF]{3,}$')

def clean_text(parts):
    """Nettoie le texte OCR en gardant les chiffres et lettres arabes (y compris أ)"""
    joined = "".join(parts)
    cleaned = ""
    for char in joined:
        if char.isdigit() or '\u0600' <= char <= '\u06FF':
            cleaned += char
    return cleaned


rtsp_url = os.getenv("CAMERA_RTSP_URL")
if not rtsp_url:
    logging.error("CAMERA_RTSP_URL n'est pas défini dans .env — arrêter le script.")
    sys.exit(1)

plate_votes = Counter()
last_sent = None

cap = cv2.VideoCapture(rtsp_url)
frame_count = 0
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", 10))

failed_read_count = 0
MAX_FAILED_READS = 10

if not cap.isOpened():
    logging.error("Impossible d'ouvrir le flux RTSP.")
    sys.exit(1)

cv2.namedWindow("OCR - Module 4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("OCR - Module 4", 480, 270)

while True:
    ret, frame = cap.read()

    if not ret or frame is None or getattr(frame, "size", 0) == 0:
        failed_read_count += 1
        if failed_read_count >= MAX_FAILED_READS:
            logging.warning("Reconnexion à la caméra...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            failed_read_count = 0
        continue

    failed_read_count = 0

    frame = cv2.resize(frame, (1280, 720))
    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        if SHOW_WINDOW:
            cv2.imshow("OCR - Module 4", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    try:
        vehicle_results = vehicle_model(frame, classes=vehicle_classes, verbose=False)
    except Exception as e:
        logging.warning(f"Erreur inference véhicule: {e}")
        continue

    nb_vehicules = len(vehicle_results[0].boxes)
    logging.debug(f"Véhicules détectés dans cette frame: {nb_vehicules}")

    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_crop = frame[y1:y2, x1:x2]
        if getattr(vehicle_crop, "size", 0) == 0:
            continue

        try:
            plate_results = plate_model(vehicle_crop, verbose=False)
        except Exception as e:
            logging.warning(f"Erreur inference plaque: {e}")
            continue

        nb_plaques = len(plate_results[0].boxes)
        logging.debug(f"  Plaques détectées dans ce véhicule: {nb_plaques}")

        for pbox in plate_results[0].boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]

            if getattr(plate_crop, "size", 0) > 0:
                text = pytesseract.image_to_string(
                    plate_crop,
                    lang='ara+eng',
                    config='--psm 7'
                )
                logging.debug(f"Texte brut Tesseract: '{text}'")
                cleaned = clean_text([text])

                if cleaned:
                    logging.info(f"Candidat OCR nettoyé: {cleaned}")
                    logging.debug(f"  → Correspondance regex: {bool(PLATE_REGEX.match(cleaned))}")

                if cleaned and PLATE_REGEX.match(cleaned):
                    plate_votes[cleaned] += 1
                    most_common, count = plate_votes.most_common(1)[0]

                    if count >= PLATE_MIN_VOTES and most_common != last_sent:
                        logging.info(f"✅ Plaque CONFIRMÉE : {most_common} ({count} lectures)")
                        try:
                            response = requests.post(
                                "http://127.0.0.1:5000/api/plates",
                                json={"plate_number": most_common},
                                timeout=5
                            )
                            logging.info(f"Envoyé au serveur : {response.status_code}")
                            last_sent = most_common
                            plate_votes.clear()
                        except requests.exceptions.ConnectionError as e:
                            logging.warning(f"⚠️ Serveur API indisponible : {e}")
                            last_sent = most_common
                            plate_votes.clear()
                        except Exception as e:
                            logging.error(f"❌ Erreur envoi : {e}")
                elif cleaned:
                    logging.debug(f"❌ Rejeté (format invalide) : {cleaned}")

            cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 0, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("OCR - Module 4", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()