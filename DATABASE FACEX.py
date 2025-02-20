import pickle
import os

# ✅ Path to saved embeddings
EMBEDDINGS_FILE = "face_embeddings.pkl"

# ✅ Load and display stored persons
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        face_db = pickle.load(f)

    if face_db:
        print("\n📂 **Stored Persons in the Database:**")
        for i, name in enumerate(face_db.keys(), 1):
            print(f"{i}. {name}")
    else:
        print("⚠️ No persons found in the database.")
else:
    print("❌ No embeddings file found.")
