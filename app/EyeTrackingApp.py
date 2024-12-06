import cv2
import mediapipe as mp
import pyautogui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QComboBox, QVBoxLayout, QWidget, QTextBrowser,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

mp_face_mesh = mp.solutions.face_mesh
pyautogui.FAILSAFE = False


class EyeTrackingThread(QThread):
    frame_updated = pyqtSignal(QImage)
    blink_detected = pyqtSignal(str)
    instruction_update = pyqtSignal(str)
    calibration_status = pyqtSignal(int)  # To update calibration progress

    def __init__(self):
        super().__init__()
        self.running = False
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        self.pupil_index = 468  # Default to Left Eye
        self.corner_points = []
        self.prev_x, self.prev_y = 0, 0
        self.screen_width, self.screen_height = pyautogui.size()

    def set_dominant_eye(self, eye_index):
        self.pupil_index = eye_index

    def capture_corner_point(self):
        if self.last_frame is not None and len(self.corner_points) < 4:
            h, w, _ = self.last_frame.shape
            pupil = self.last_landmarks.landmark[self.pupil_index]
            pupil_x, pupil_y = int(pupil.x * w), int(pupil.y * h)
            self.corner_points.append((pupil_x, pupil_y))
            self.instruction_update.emit(f"Corner {len(self.corner_points)} captured.")
            self.calibration_status.emit(len(self.corner_points))

    def run(self):
        self.running = True
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            self.last_frame = frame  # Store the last frame for calibration
            self.last_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

            h, w, _ = frame.shape
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pupil = face_landmarks.landmark[self.pupil_index]
                    pupil_x, pupil_y = int(pupil.x * w), int(pupil.y * h)
                    cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)

            if len(self.corner_points) < 4:
                label = f"Look at corner {len(self.corner_points) + 1} and press 'Calibrate'"
                self.instruction_update.emit(label)

            # Convert frame to QImage for GUI display
            qt_frame = QImage(frame.data, w, h, QImage.Format_RGB888)
            self.frame_updated.emit(qt_frame)

    def stop(self):
        self.running = False
        self.cap.release()


class EyeTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking GUI")
        self.setGeometry(100, 100, 800, 600)

        self.thread = EyeTrackingThread()
        self.thread.frame_updated.connect(self.update_frame)
        self.thread.blink_detected.connect(self.display_blink_message)
        self.thread.instruction_update.connect(self.update_instructions)
        self.thread.calibration_status.connect(self.update_calibration_progress)

        # GUI Elements
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.start_button = QPushButton("Start Tracking", self)
        self.start_button.clicked.connect(self.start_tracking)

        self.stop_button = QPushButton("Stop Tracking", self)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)

        self.calibrate_button = QPushButton("Calibrate", self)
        self.calibrate_button.clicked.connect(self.calibrate_corner)

        self.eye_selector = QComboBox(self)
        self.eye_selector.addItems(["Left Eye", "Right Eye"])
        self.eye_selector.currentIndexChanged.connect(self.change_dominant_eye)

        self.instructions = QTextBrowser(self)
        self.instructions.setText("Instructions will appear here.")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.calibrate_button)
        layout.addWidget(self.eye_selector)
        layout.addWidget(self.instructions)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_tracking(self):
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.instructions.setText("Look at the camera and follow the instructions.")

    def stop_tracking(self):
        self.thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.instructions.setText("Tracking stopped.")

    def change_dominant_eye(self, index):
        self.thread.set_dominant_eye(468 if index == 0 else 473)

    def calibrate_corner(self):
        self.thread.capture_corner_point()

    def update_frame(self, qt_frame):
        pixmap = QPixmap.fromImage(qt_frame)
        self.video_label.setPixmap(pixmap)

    def display_blink_message(self, message):
        self.instructions.append(message)

    def update_instructions(self, instruction):
        self.instructions.setText(instruction)

    def update_calibration_progress(self, points_captured):
        if points_captured == 4:
            self.instructions.setText("Calibration complete! Tracking active.")


if __name__ == "__main__":
    app = QApplication([])
    window = EyeTrackingApp()
    window.show()
    app.exec()

