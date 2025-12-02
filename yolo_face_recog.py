import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================
# 1. LOAD YOLOv8-FACE DETECTOR
# ============================
print("🚀 Loading YOLOv8-face model...")
detector = YOLO("yolov8-face.pt")
print("✔ YOLOv8-face loaded")


# ================================
# 2. LOAD INSIGHTFACE RECOGNIZER
# ================================
print("🚀 Loading InsightFace (buffalo_l)...")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("✔ InsightFace loaded. Using GPU if available.")


# ================================
# 3. LOAD SINGLE REFERENCE IMAGE
# ================================
REFERENCE_PATH = "tigerz.jpg"

print(f"📂 Loading reference face from {REFERENCE_PATH}...")

img = cv2.imread(REFERENCE_PATH)
if img is None:
    raise FileNotFoundError("❌ tigerz.jpg NOT found!")

faces = face_app.get(img)
if len(faces) == 0:
    raise RuntimeError("❌ No face detected in tigerz.jpg!")

reference_embedding = faces[0].normed_embedding.copy()
print("✔ Reference embedding loaded:", reference_embedding.shape)


# =============================
# 4. COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b)


THRESHOLD = 0.36


# ======================
# 5. CAMERA LOOP
# ======================
cap = cv2.VideoCapture(0)
print("\n🎥 Starting webcam... Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Use InsightFace on full frame
        detected = face_app.get(frame)

        best_sim = 0
        label = "Unknown"

        for f in detected:
            sim = cosine_similarity(f.normed_embedding, reference_embedding)
            print(f"[DEBUG] similarity={sim:.4f}  threshold={THRESHOLD}")

            if sim > best_sim:
                best_sim = sim

            if sim >= THRESHOLD:
                label = "TigerZ"
                break

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({best_sim:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
