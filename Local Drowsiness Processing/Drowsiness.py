from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import requests
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# settings
thresh = 0.20
frame_check = 30
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Landmark
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Camera Stream
cap = cv2.VideoCapture("http://192.168.0.170:81/stream")

flag = 0
alert_active = False
frame_counter = 0    # logging periodik

def send_signal(url):
    try:
        requests.get(url, timeout=0.3)
    except:
        pass

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame kosong dari kamera!")
        continue

    frame_counter += 1

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    if len(subjects) == 0:
        if alert_active:  
            print("[INFO] Wajah hilang → Matikan alert")
            send_signal("http://192.168.0.138/off")
            alert_active = False
        flag = 0
        continue

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Logging EAR setiap 10 frame
        if frame_counter % 10 == 0:
            print(f"[EAR] {ear:.3f}")

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1

            # Logging saat flag berubah → tidak spam
            print(f"[DROWSY] Flag = {flag}")

            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not alert_active:
                    print("[ALERT] Mengantuk! → Kirim ON")
                    send_signal("http://192.168.0.138/on")
                    alert_active = True

        else:
            if flag != 0:
                print(f"[RESET] Flag kembali ke 0 (ear normal: {ear:.3f})")
            flag = 0

            if alert_active:
                print("[ALERT] Mata terbuka → Kirim OFF")
                send_signal("http://192.168.0.138/off")
                alert_active = False

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()