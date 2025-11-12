from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import os
import time
from threading import Thread

# === Step 1: Load reference faces ===
REFERENCE_IMAGES = {
    "Suspect1": "Neo.jpg"
}

known_encodings, known_names = [], []
for name, img_path in REFERENCE_IMAGES.items():
    if not os.path.exists(img_path):
        print(f"❌ Missing {img_path}")
        continue
    img = face_recognition.load_image_file(img_path)
    encs = face_recognition.face_encodings(img)
    if len(encs) == 0:
        print(f"⚠️ No face found in {img_path}")
        continue
    known_encodings.append(encs[0])
    known_names.append(name)
    print(f"✅ Loaded {name}")

if not known_encodings:
    print("❌ No reference faces loaded. Exiting.")
    exit()

# === Step 2: Load YOLOv8-face model ===
MODEL_PATH = "yolov8-face.pt"
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found. Exiting.")
    exit()
model = YOLO(MODEL_PATH)

# === Step 3: Detect available local camera ===
def get_available_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            print(f"✅ Found camera at index {i}")
            return i
        cap.release()
    print("❌ No local camera detected")
    return -1

camera_index = get_available_camera()
if camera_index == -1:
    exit()

# === Step 4: Threaded camera class ===
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            print("⚠️ Warning: First frame not grabbed")
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print("⚠️ Frame grab failed. Stopping camera.")
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# === Step 5: Start threaded video stream ===
print("📸 Starting threaded camera...")
camera = CameraStream(camera_index)
time.sleep(1.0)  # Allow camera to warm up

prev_time = time.time()
fps_smooth = 0

# Create resizable window with double size
cv2.namedWindow("🚀 Real-Time YOLOv8n-face", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🚀 Real-Time YOLOv8n-face", 960, 540)  # double size

while True:
    frame = camera.read()
    if frame is None:
        continue

    # Resize frame for YOLO detection (speed)
    small_frame = cv2.resize(frame, (640, 360))

    # YOLO face detection
    results = model.predict(small_frame, conf=0.45, imgsz=320, verbose=False)
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            # Scale coordinates to original frame size
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_crop)

            label = "Unknown"
            color = (0, 255, 255)
            if len(encs) > 0:
                face_encoding = encs[0]
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if True in matches:
                    best_idx = np.argmin(face_distances)
                    name = known_names[best_idx]
                    conf = round((1 - face_distances[best_idx]) * 100, 1)
                    label = f"{name} ({conf}%)"
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Smoothed FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps_smooth)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Resize frame for display (double size)
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow("🚀 Real-Time YOLOv8n-face", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
