import cv2
import numpy as np
import os
import pickle
import insightface
from insightface.app import FaceAnalysis

# ✅ Local model path
MODEL_PATH = r"C:\Users\yk040\Downloads\buffalo_l"

# ✅ Load Face Analysis with predefined detection size
app = FaceAnalysis(name=MODEL_PATH, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))  # Set detection size globally

# ✅ Path to save embeddings
EMBEDDINGS_FILE = "face_embeddings.pkl"

# ✅ Load existing embeddings
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# ✅ Capture multiple images
def capture_webcam(num_images=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return []

    images = []
    print(f"📸 Press SPACE to capture {num_images} images.")

    while len(images) < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == 32:  # Press SPACE to capture
            images.append(frame)
            print(f"✅ Captured {len(images)}/{num_images}")

    cap.release()
    cv2.destroyAllWindows()
    return images

# ✅ Train with captured images
def train_on_webcam():
    person_name = input("Enter person's name: ").strip()
    if not person_name:
        print("❌ Name cannot be empty.")
        return

    images = capture_webcam(num_images=10)  # Capture 10 images
    if not images:
        print("❌ No images captured.")
        return

    embeddings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        faces = app.get(img)  # ✅ FIXED: Removed det_size argument

        if len(faces) == 0:
            print("❌ No face detected in one of the images.")
            continue

        face = faces[0]  
        normed_embedding = face.normed_embedding / np.linalg.norm(face.normed_embedding)  # Normalize
        embeddings.append(normed_embedding)

    if not embeddings:
        print("❌ No valid faces detected.")
        return

    # ✅ Compute average embedding
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)  # Normalize again

    # ✅ Save embedding
    face_db[person_name] = mean_embedding
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(face_db, f)

    print(f"✅ {person_name} added to the database!")

# Run training
train_on_webcam()
