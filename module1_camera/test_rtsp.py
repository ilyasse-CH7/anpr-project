import cv2
import os
from dotenv import load_dotenv

load_dotenv()
rtsp_url = os.getenv("CAMERA_RTSP_URL")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Erreur : impossible de se connecter à la caméra IP")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : flux non reçu")
        break

    cv2.imshow("Camera IP - Test Module 1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()