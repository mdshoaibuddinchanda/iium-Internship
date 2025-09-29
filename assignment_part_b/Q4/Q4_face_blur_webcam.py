"""
Ultimate Face Masked Blurring (YOLOv8x + MediaPipe FaceMesh)
------------------------------------------------------------
- Uses YOLOv8x (local) for fast & robust face detection.
- Uses MediaPipe FaceMesh to get 468 facial landmarks.
- Builds a precise face polygon mask (jawline to forehead).
- Blurs only the face area (not hair/background).
- Works for Webcam & Video input.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress absl + TensorFlow Lite logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Redirect unwanted stderr messages to null
sys.stderr = open(os.devnull, "w")


import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
  # Suppress TensorFlow Lite logs


# ---------- Config ----------
MODEL_PATH = "assignment_part_b/Q4/yolov8m-face.pt"

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    exit()

# Load YOLO model
model = YOLO(MODEL_PATH)

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,  # includes iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Input ----------
choice = input("Do you want to use Webcam or Video? (w/v): ").strip().lower()

if choice == "w":
    cap = cv2.VideoCapture(0)
    output_video = "assignment_part_b\result\Q4\output_blurred_webcam.mp4"
elif choice == "v":
    video_path = input("Enter full path of video file: ").strip()
    cap = cv2.VideoCapture(video_path)
    output_video = "assignment_part_b\result\Q4\output_blurred_video.mp4"
else:
    print("Invalid choice. Exiting...")
    exit()

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4)))
)

print("\n[INFO] Processing... Press ESC to stop.\n")

# ---------- Processing ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # Detect faces with YOLO
    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Run Face Mesh
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_face)

            if results_mesh.multi_face_landmarks:
                for landmarks in results_mesh.multi_face_landmarks:
                    points = []
                    for lm in landmarks.landmark:
                        px, py = int(lm.x * (x2 - x1)) + x1, int(lm.y * (y2 - y1)) + y1
                        points.append([px, py])

                    # Convert to numpy array
                    points = np.array(points, dtype=np.int32)

                    # Fill convex hull around face
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(mask, hull, 255)

    # Apply blur only on masked face regions
    if mask.sum() > 0:
        blurred = cv2.GaussianBlur(frame, (99, 99), 30)
        face_only = cv2.bitwise_and(blurred, blurred, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        output = cv2.add(background, face_only)
    else:
        output = frame

    cv2.imshow("Face Blur", output)
    out.write(output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Done! Blurred video saved as: {output_video}")
