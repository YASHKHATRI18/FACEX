import cv2
import numpy as np
import os
import pickle
import insightface
from insightface.app import FaceAnalysis

# ✅ Local model path
MODEL_PATH = r"C:\Users\yk040\Downloads\buffalo_l"

# ✅ Load Face Analysis
app = FaceAnalysis(name=MODEL_PATH, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# ✅ Load saved embeddings
EMBEDDINGS_FILE = "face_embeddings.pkl"

if not os.path.exists(EMBEDDINGS_FILE):
    print("❌ No saved embeddings found.")
    exit()

with open(EMBEDDINGS_FILE, "rb") as f:
    face_db = pickle.load(f)

# ✅ Cosine similarity function
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# ✅ Matching threshold
THRESHOLD = 0.6 

# ✅ Capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("📷 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    faces = app.get(img)

    match_name = "Unknown"

    if len(faces) > 0:
        face = faces[0]
        test_embedding = face.normed_embedding
        test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize

        best_match = None
        best_score = -1

        for name, stored_embedding in face_db.items():
            similarity = cosine_similarity(test_embedding, stored_embedding)

            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_score >= THRESHOLD:
            match_name = best_match

        # ✅ Get face bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)

        # ✅ Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ✅ Display the name on the webcam feed
        cv2.putText(frame, match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
