import cv2
import os
import urllib.request
from ultralytics import YOLO


# Modèle véhicule (Module 2)
vehicle_model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]

# Modèle plaque pré-entraîné (Module 3) - téléchargé automatiquement


model_url = "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt"
model_path = "models/license-plate-finetune-v1n.pt"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Téléchargement du modèle plaque...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Téléchargement terminé")

plate_model = YOLO(model_path)
print("Modèle plaque chargé avec succès")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_results = vehicle_model(frame, classes=vehicle_classes, verbose=False)

    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            continue

        plate_results = plate_model(vehicle_crop, verbose=False)

        for pbox in plate_results[0].boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 0, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Detection Plaque - Module 3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()