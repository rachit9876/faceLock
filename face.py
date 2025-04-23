import cv2
import time
import os
import mediapipe as mp
import traceback
import face_recognition
import numpy as np

# Constants
EYE_CLOSED_SECONDS = 10  # Repurposed as face-not-detected duration
# CLOSED_EYE_RATIO_THRESHOLD = 0.25  # Commented out as eye detection is disabled

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)

# LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Commented out
# RIGHT_EYE_LANDMARKS = [362, 385 ehe, 387, 263, 373, 380]  # Commented out

closed_start_time = None
locked = False
countdown_text = ""
nose_connections = list(mp_face_mesh.FACEMESH_NOSE)
reference_encoding = None  # Stores the locked face encoding
is_face_locked = False  # Tracks if a face is locked

# Capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    try:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        h, w = frame.shape[:2]

        # Process face detection and mesh on full frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        face_detected = False
        status_text = "Rachit Detected" if is_face_locked else "Press 'l' to lock face"
        status_color = (0, 255, 0)

        try:
            if face_results.detections:
                if is_face_locked:
                    # Track only the locked face
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces([reference_encoding], encoding, tolerance=0.5)
                        if matches[0]:
                            face_detected = True
                            x1, y1 = left, top
                            bw, bh = right - left, bottom - top
                            if x1 >= 0 and y1 >= 0 and x1 + bw <= w and y1 + bh <= h:
                                cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                else:
                    # Normal detection mode
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        bw = int(bboxC.width * w)
                        bh = int(bboxC.height * h)
                        if x1 >= 0 and y1 >= 0 and x1 + bw <= w and y1 + bh <= h:
                            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                            face_detected = True
                            # Capture face encoding if locking
                            if cv2.waitKey(1) & 0xFF == ord('l'):
                                face_locations = face_recognition.face_locations(rgb_frame)
                                if face_locations:
                                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                    if face_encodings:
                                        reference_encoding = face_encodings[0]
                                        is_face_locked = True
                                        status_text = "Face Locked"
        except Exception as e:
            print(f"Error in face detection processing: {e}")
            traceback.print_exc()

        try:
            if mesh_results.multi_face_landmarks and face_detected:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Draw full mesh
                    for connection in mp_face_mesh.FACEMESH_TESSELATION:
                        start_idx, end_idx = connection
                        start = face_landmarks.landmark[start_idx]
                        end = face_landmarks.landmark[end_idx]
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        if (x1 >= 0 and y1 >= 0 and x1 < w and y1 < h and
                            x2 >= 0 and y2 >= 0 and x2 < w and y2 < h):
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

                    # Draw nose mesh in green
                    for connection in nose_connections:
                        start_idx, end_idx = connection
                        start = face_landmarks.landmark[start_idx]
                        end = face_landmarks.landmark[end_idx]
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        if (x1 >= 0 and y1 >= 0 and x1 < w and y1 < h and
                            x2 >= 0 and y2 >= 0 and x2 < w and y2 < h):
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in mesh drawing: {e}")
            traceback.print_exc()

        # Unlock face if 'u' is pressed
        if cv2.waitKey(1) & 0xFF == ord('u'):
            is_face_locked = False
            reference_encoding = None
            status_text = "Face Unlocked"

        # Countdown logic
        if not face_detected:
            if closed_start_time is None:
                closed_start_time = time.time()
            elapsed = time.time() - closed_start_time

            countdown = EYE_CLOSED_SECONDS - int(elapsed)
            if countdown > 0:
                countdown_text = f"System Off in {countdown}s"
                status_text = "System Off"
                status_color = (0, 0, 255)
            else:
                status_text = "System Off"
                status_color = (0, 0, 255)
                countdown_text = "Shutting down..."
                if not locked:
                    os.system("rundll32.exe user32.dll,LockWorkStation")
                    locked = True
        else:
            closed_start_time = None
            locked = False
            countdown_text = ""

        # Display status and countdown
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        if countdown_text:
            cv2.putText(frame, countdown_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Neural Face Lock System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Main loop error: {e}")
        traceback.print_exc()
        continue

cap.release()
cv2.destroyAllWindows()