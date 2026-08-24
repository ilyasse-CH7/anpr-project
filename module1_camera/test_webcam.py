import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : image non reçue")
        break

    cv2.imshow("Webcam - Test Module 1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()