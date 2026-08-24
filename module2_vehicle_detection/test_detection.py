import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]
class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : image non reçue")
        break

    results = model(frame, classes=vehicle_classes, verbose=False)

    # Parcourir chaque détection et afficher ses coordonnées
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(f"{class_names[class_id]} détecté : ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)}) | confiance: {confidence:.2f}")

    annotated_frame = results[0].plot()
    cv2.imshow("Detection Vehicule - Module 2", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()