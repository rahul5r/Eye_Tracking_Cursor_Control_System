from flask import Flask, render_template, Response
import cv2
import pyautogui
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

app = Flask(__name__)
camera = cv2.VideoCapture(0)
tracking_active = False

LEFT_EYE_PUPIL_INDEX = 468
RIGHT_EYE_PUPIL_INDEX = 473
eye_landmarks = {
    'Top Left Eyelid': 159, 'Bottom Left Eyelid': 145,
    'Top Right Eyelid': 386, 'Bottom Right Eyelid': 374,
}
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

# Bounding box dimensions
bounding_box_width = screen_width * 0.1
bounding_box_height = screen_height * 0.1

prev_x, prev_y = 0, 0
PUPIL_INDEX = RIGHT_EYE_PUPIL_INDEX  # Default to right eye for demonstration

def map_range(value, from_min, from_max, to_min, to_max):
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

def resize_frame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def check_blink(landmarks):
    """
    Detect blink by checking the vertical distance between the eyelids.
    """
    left_eye = abs(landmarks[0][1] - landmarks[1][1])
    right_eye = abs(landmarks[2][1] - landmarks[3][1])
    
    # Thresholds for blinking
    blink_threshold = 20
    if left_eye < blink_threshold and right_eye < blink_threshold:
        return False  # Both eyes closed
    elif left_eye < blink_threshold and PUPIL_INDEX == RIGHT_EYE_PUPIL_INDEX:
        return True  # Right eye controlling cursor, left eye used for clicking
    elif right_eye < blink_threshold and PUPIL_INDEX == LEFT_EYE_PUPIL_INDEX:
        return True  # Left eye controlling cursor, right eye used for clicking
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global tracking_active, prev_x, prev_y
    while tracking_active:
        success, frame = camera.read()
        if not success:
            break
        
        image = cv2.flip(frame, 1)
        image = resize_frame(image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)

        h, w, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pupil = face_landmarks.landmark[PUPIL_INDEX]
                pupil_x = int(pupil.x * w)
                pupil_y = int(pupil.y * h)

                cv2.circle(image, (pupil_x, pupil_y), 5, (0, 255, 0), -1)

                # Blink Detection Part
                landmarks = []
                for eye_part, index in eye_landmarks.items():
                    lm = face_landmarks.landmark[index]
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append([x, y])
                    if PUPIL_INDEX == RIGHT_EYE_PUPIL_INDEX:
                        if eye_part == 'Top Left Eyelid' or eye_part == 'Bottom Left Eyelid':
                            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                    else:
                        if eye_part == 'Top Right Eyelid' or eye_part == 'Bottom Right Eyelid':
                            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

                blink = check_blink(landmarks)
                if blink:
                    print("Blink Detected!")
                    pyautogui.click()
                    cv2.putText(image, "Blink Detected! Clicking...", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # Calculate dynamic bounding box
                box_top_left = (int(w / 2 - bounding_box_width / 2), int(h / 2 - bounding_box_height / 2))
                box_bottom_right = (int(w / 2 + bounding_box_width / 2), int(h / 2 + bounding_box_height / 2))

                cv2.rectangle(image, box_top_left, box_bottom_right, (255, 0, 0), 2)

                if box_top_left[0] <= pupil_x <= box_bottom_right[0] and box_top_left[1] <= pupil_y <= box_bottom_right[1]:
                    mapped_x = map_range(pupil_x, box_top_left[0], box_bottom_right[0], 0, screen_width)
                    mapped_y = map_range(pupil_y, box_top_left[1], box_bottom_right[1], 0, screen_height)

                    alpha = 0.5  # Adjust smoothing factor for faster cursor response
                    smooth_x = alpha * mapped_x + (1 - alpha) * prev_x
                    smooth_y = alpha * mapped_y + (1 - alpha) * prev_y

                    try:
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    except pyautogui.FailSafeException:
                        pyautogui.moveTo(0, 0)

                    prev_x, prev_y = smooth_x, smooth_y

        _, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global tracking_active
    if tracking_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Tracking not active", 200

@app.route('/start', methods=['POST'])
def start_tracking():
    global tracking_active, camera
    if not camera.isOpened():  # Check if the camera is not already open
        camera = cv2.VideoCapture(0)
    tracking_active = True
    return "Tracking started", 200

@app.route('/stop', methods=['POST'])
def stop_tracking():
    global tracking_active, camera
    tracking_active = False
    if camera.isOpened():
        camera.release()
    return "Tracking stopped", 200

if __name__ == '__main__':
    app.run(debug=True)