import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Important landmarks
LEFT_EYE_PUPIL_INDEX = 468
RIGHT_EYE_PUPIL_INDEX = 473
eye_landmarks = {
    'Top Left Eyelid': 159, 'Bottom Left Eyelid': 145,
    'Top Right Eyelid': 386, 'Bottom Right Eyelid': 374,
}
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

corner_points = []
corner_labels = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

# Default Corner Points
# corner_points = [(331, 298), (479, 294), (348, 356), (431, 353)]      # Left Eye
# corner_points = [(492, 365), (633, 375), (557, 411), (645, 422)]      # Right Eye

def get_domnant_eye():
    print("Enter your Domant Eye : ")
    print("1. Left Eye (Mouse Control with Left Eye and Clicking with Right Eye)")
    print("2. Right Eye (Mouse Control with Right Eye and Clicking with Left Eye)")
    choice = int(input("Enter your choice (1 or 2) : "))
    
    if choice == 1:
        return LEFT_EYE_PUPIL_INDEX
    elif choice == 2:
        return RIGHT_EYE_PUPIL_INDEX

def check_blink(landmarks):
    left_eye = abs(landmarks[0][1] - landmarks[1][1])
    right_eye = abs(landmarks[2][1] - landmarks[3][1])
    
    if left_eye < 4 and right_eye < 4:
        return False
    elif left_eye < 4 and PUPIL_INDEX == RIGHT_EYE_PUPIL_INDEX:
        return True
    elif right_eye < 4 and PUPIL_INDEX == LEFT_EYE_PUPIL_INDEX:
        return True
    else:
        return None

def map_range(value, from_min, from_max, to_min, to_max):
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

def capture_eye_position(pupil_x, pupil_y):
    if len(corner_points) < 4:
        corner_points.append((pupil_x, pupil_y))
        print(f"Captured corner point {len(corner_points)}: {pupil_x}, {pupil_y}")

def resizeFrame(frame, scale=1.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)

prev_x, prev_y = 0,0
PUPIL_INDEX = get_domnant_eye()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image = resizeFrame(image)
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

            for eye_part, indices in eye_landmarks.items():
                lm = face_landmarks.landmark[indices]
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append([x,y])
                if PUPIL_INDEX == RIGHT_EYE_PUPIL_INDEX:
                    if eye_part == 'Top Left Eyelid' or eye_part == 'Bottom Left Eyelid':
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                else:
                    if eye_part == 'Top Right Eyelid' or eye_part == 'Bottom Right Eyelid':
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

            blink = check_blink(landmarks)
            if blink:
                print("Blink Detected..")
                pyautogui.click()
                cv2.putText(image, "Blink Detected.. ", (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                capture_eye_position(pupil_x, pupil_y)

            if len(corner_points) == 4:
                min_x = min([pt[0] for pt in corner_points])
                max_x = max([pt[0] for pt in corner_points])
                min_y = min([pt[1] for pt in corner_points])
                max_y = max([pt[1] for pt in corner_points])

                box_top_left = (min_x, min_y)
                box_bottom_right = (max_x, max_y)

                cv2.rectangle(image, box_top_left, box_bottom_right, (255, 0, 0), 2)

                if min_x <= pupil_x <= max_x and min_y <= pupil_y <= max_y:
                    mapped_x = map_range(pupil_x, min_x, max_x, 0, screen_width)
                    mapped_y = map_range(pupil_y, min_y, max_y, 0, screen_height)
                    
                    alpha = 0.5  # smoothing factor, smaller values mean more smoothing
                    smooth_x = alpha * mapped_x + (1 - alpha) * prev_x
                    smooth_y = alpha * mapped_y + (1 - alpha) * prev_y
                    
                    try:
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    except pyautogui.FailSafeException:
                        pyautogui.moveTo(0, 0)
                    prev_x, prev_y = smooth_x, smooth_y

    if len(corner_points) < 4:
        cv2.putText(image, f'Look at {corner_labels[len(corner_points)]} corner and press "c"', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Eye Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
