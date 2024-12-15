import time
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import pyautogui
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

app = Flask(__name__)
camera = None

tracking_active = False
dominant_eye = None
bounding_box_ratio = 0.1
corner_points = []
corner_labels = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

LEFT_EYE_PUPIL_INDEX = 468
RIGHT_EYE_PUPIL_INDEX = 473
eye_landmarks = {
    'Top Left Eyelid': 159, 'Bottom Left Eyelid': 145,
    'Top Right Eyelid': 386, 'Bottom Right Eyelid': 374,
}
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

prev_x, prev_y = 0, 0


@app.route('/')
def index():
    return render_template('calibration.html')

@app.route('/calibration', methods=['GET', 'POST'])
def calibration():
    global dominant_eye, corner_points, PUPIL_INDEX
    
    corner_points = []
    if request.method == 'POST':
        if 'dominant_eye' in request.form:
            dominant_eye = request.form['dominant_eye']
            if str(dominant_eye) == "left":
                print("left eye dominant")
                PUPIL_INDEX = LEFT_EYE_PUPIL_INDEX
            else:
                print("right eye dominant")
                PUPIL_INDEX = RIGHT_EYE_PUPIL_INDEX
            return redirect(url_for('calibration_corners'))
    return render_template('calibration.html')

cap = cv2.VideoCapture(0)

def generate_video_stream():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed_corner')
def video_feed_corner():
    return Response(generate_video_stream_with_pupil(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_stream_with_pupil():
    global cap, face_mesh, PUPIL_INDEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pupil = face_landmarks.landmark[PUPIL_INDEX]
                pupil_x = int(pupil.x * w)
                pupil_y = int(pupil.y * h)


                cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/calibration/corners', methods=['GET', 'POST'])
def calibration_corners():
    global corner_points, cap, PUPIL_INDEX
    print(f"Using PUPIL_INDEX: {PUPIL_INDEX}")

    if request.method == 'POST':
        ret, frame = cap.read()

        if not ret:
            return "Error: Could not capture frame", 500
        
        frame = cv2.flip(frame, 1)
        frame = resize_frame(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)

        h, w, _ = frame.shape
        print("Landmarks detected:", bool(results.multi_face_landmarks))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pupil = face_landmarks.landmark[PUPIL_INDEX]
                pupil_x = int(pupil.x * w)
                pupil_y = int(pupil.y * h)

                cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)

        center_x, center_y = w // 2, h // 2
        corner_points.append((center_x, center_y))

        if len(corner_points) == 4:
            print("Calibration points:", corner_points)
            cap.release()
            return redirect(url_for('tracking'))
    corner_visible = corner_labels[len(corner_points)] 
    if len(corner_points) <= 4:
        return render_template('calibration_corners.html', 
                           corner_label=corner_labels[len(corner_points)], 
                           corner_visible=corner_visible)
    else:
        return redirect(url_for('tracking'))

@app.route('/tracking')
def tracking():
    update_bounding_box_ratio()
    return render_template('tracking.html')


@app.route('/start', methods=['POST'])
def start_tracking():
    global tracking_active, camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    tracking_active = True
    print("Tracking Status : ", tracking_active)
    
    return "Tracking started", 200

@app.route('/stop', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = False
    print("Tracking Status : ", tracking_active)
    return "Tracking stopped", 200

@app.route('/video_feed')
def video_feed():
    global tracking_active
    print("Tracking Status : ",tracking_active)
    if tracking_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Tracking not active", 200

def map_range(value, from_min, from_max, to_min, to_max):
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

def resize_frame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def check_blink(landmarks):
    left_eye = abs(landmarks[0][1] - landmarks[1][1])
    right_eye = abs(landmarks[2][1] - landmarks[3][1])
    
    # Threshold for blinking
    blink_threshold = 15
    if left_eye < blink_threshold and right_eye < blink_threshold:
        return False  # Both eyes closed
    elif left_eye < blink_threshold and PUPIL_INDEX == RIGHT_EYE_PUPIL_INDEX:
        return True  # Right eye controlling cursor, left eye used for clicking
    elif right_eye < blink_threshold and PUPIL_INDEX == LEFT_EYE_PUPIL_INDEX:
        return True  # Left eye controlling cursor, right eye used for clicking
    else:
        return None

def update_bounding_box_ratio():
    global corner_points, bounding_box_ratio

    if len(corner_points) != 4:
        print("Error: Insufficient corner points for bounding box calculation.") 
        return

    x_coords = [pt[0] for pt in corner_points]
    y_coords = [pt[1] for pt in corner_points]

    # Calculate bounding box dimensions
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    box_width = max_x - min_x
    box_height = max_y - min_y

    # Update the bounding box ratio relative to the screen dimensions
    screen_width, screen_height = pyautogui.size()
    width_ratio = box_width / screen_width
    height_ratio = box_height / screen_height

    # Use the larger of the two ratios for consistency
    bounding_box_ratio = max(width_ratio, height_ratio)
    if bounding_box_ratio < 0.1:
        bounding_box_ratio = 0.1
    if bounding_box_ratio > 0.7:
        bounding_box_ratio = 0.7
    print(f"Updated bounding_box_ratio: {bounding_box_ratio}")

def generate_frames():
    global tracking_active, prev_x, prev_y, bounding_box_ratio
    print(bounding_box_ratio)

    # Bounding box dimensions
    bounding_box_width = screen_width * bounding_box_ratio
    bounding_box_height = screen_height * bounding_box_ratio

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

if __name__ == '__main__':
    app.run(debug=True)