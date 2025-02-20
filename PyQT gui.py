import sys
import cv2
import os
import sqlite3
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QMessageBox,
    QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Global variables
DATASET_PATH = "./dataset"  # Path to store captured images
MODEL_PATH = "./model/buffalo_l"  # Pretrained model
EMBEDDINGS_PATH = "./embeddings.pkl"
DATABASE_PATH = "./attendance.db"
SIMILARITY_THRESHOLD = 0.5

# Initialize FaceAnalysis
app_insightface = FaceAnalysis(name=MODEL_PATH)
app_insightface.prepare(ctx_id=0)  # Use GPU (ctx_id=0) or CPU (ctx_id=-1)

# SQLite Database Initialization
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    date TEXT,
                    time TEXT
                )''')
    conn.commit()
    conn.close()

init_database()

class HomeWindow(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 400, 300)

        # Buttons
        scan_button = QPushButton("Scan", self)
        scan_button.clicked.connect(self.open_scan_window)

        capture_button = QPushButton("Capture New Images", self)
        capture_button.clicked.connect(self.open_capture_window)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(scan_button)
        layout.addWidget(capture_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_scan_window(self):
        self.scan_window = ScanWindow()
        self.scan_window.show()
        self.close()

    def open_capture_window(self):
        self.capture_window = CaptureWindow()
        self.capture_window.show()
        self.close()

class ScanWindow(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Scan")
        self.setGeometry(100, 100, 800, 600)

        # Video feed
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        # Back button
        back_button = QPushButton("Back to Home", self)
        back_button.clicked.connect(self.go_back)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(back_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Load embeddings
        with open(EMBEDDINGS_PATH, "rb") as f:
            self.embeddings_dict = pickle.load(f)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app_insightface.get(rgb_frame)

            if faces:
                for face in faces:
                    embedding = face.embedding
                    max_similarity = 0
                    recognized_name = "Unknown"

                    for person_name, stored_embedding in self.embeddings_dict.items():
                        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                        if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                            max_similarity = similarity
                            recognized_name = person_name

                    # Draw bounding box and name
                    bbox = [int(coord) for coord in face.bbox]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Welcome {recognized_name}!", (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    if recognized_name != "Unknown":
                        self.log_entry(recognized_name)

            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(image))

    def log_entry(self, name):
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        # Check if the person has logged in within the last 2 minutes
        c.execute("SELECT MAX(time) FROM attendance WHERE name = ? AND date = ?", (name, datetime.date.today().strftime('%Y-%m-%d')))
        last_entry_time = c.fetchone()[0]

        if last_entry_time:
            last_entry_time = datetime.datetime.strptime(last_entry_time, '%H:%M:%S')
            if (datetime.datetime.now() - last_entry_time).total_seconds() < 120:
                return

        # Log entry
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", 
                  (name, datetime.date.today().strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%H:%M:%S')))
        conn.commit()
        conn.close()

    def go_back(self):
        self.cap.release()
        self.timer.stop()
        self.home_window = HomeWindow()
        self.home_window.show()
        self.close()

class CaptureWindow(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Capture New Images")
        self.setGeometry(100, 100, 800, 600)

        self.image_count = 0
        self.person_name = ""

        # Widgets
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter name")

        self.capture_button = QPushButton("Capture Image", self)
        self.capture_button.clicked.connect(self.capture_image)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)

        self.back_button = QPushButton("Back to Home", self)
        self.back_button.clicked.connect(self.go_back)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.back_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(image))

    def capture_image(self):
        if not self.person_name:
            self.person_name = self.name_input.text()

            if not self.person_name:
                QMessageBox.warning(self, "Error", "Please enter a name before capturing images!")
                return

        ret, frame = self.cap.read()
        if ret:
            person_folder = os.path.join(DATASET_PATH, self.person_name)
            os.makedirs(person_folder, exist_ok=True)

            image_path = os.path.join(person_folder, f"image_{self.image_count + 1}.jpg")
            cv2.imwrite(image_path, frame)
            self.image_count += 1

            QMessageBox.information(self, "Captured", f"Image {self.image_count} captured successfully!")

            if self.image_count == 10:
                self.capture_button.setEnabled(False)
                self.train_button.setEnabled(True)

    def train_model(self):
        # Call training script
        QMessageBox.information(self, "Training", "Model training started!")

        os.system("python train_model.py")  # Replace with actual training script

        QMessageBox.information(self, "Training", "Model trained successfully!")
        self.go_back()

    def go_back(self):
        self.cap.release()
        self.timer.stop()
        self.home_window = HomeWindow()
        self.home_window.show()
        self.close()

if _name_ == "_main_":
    app = QApplication(sys.argv)
    main_window = HomeWindow()
    main_window.show()
    sys.exit(app.exec_())
