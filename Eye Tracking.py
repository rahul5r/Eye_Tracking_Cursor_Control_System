import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.)

cap = cv2.VideoCapture(0)

LEFT_EYE_PUPIL_INDEX = 468
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

# Default Corner Points
corner_points = [(331, 298), (479, 294), (348, 356), (431, 353)]

# corner_points = []
corner_labels = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

def map_range(value, from_min, from_max, to_min, to_max):
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


def capture_eye_position(pupil_x, pupil_y):
    if len(corner_points) < 4:
        corner_points.append((pupil_x, pupil_y))
        print(f"Captured corner point {len(corner_points)}: {pupil_x}, {pupil_y}")

def check_box_size(top_left, bottom_right):
    # print(top_left, top_right)
    pass


def resizeFrame(frame, scale=1.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)

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
            left_pupil = face_landmarks.landmark[LEFT_EYE_PUPIL_INDEX]

            left_pupil_x = int(left_pupil.x * w)
            left_pupil_y = int(left_pupil.y * h)

            cv2.circle(image, (left_pupil_x, left_pupil_y), 5, (0, 255, 0), -1)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                capture_eye_position(left_pupil_x, left_pupil_y)

            if len(corner_points) == 4:
                min_x = min([pt[0] for pt in corner_points])
                max_x = max([pt[0] for pt in corner_points])
                min_y = min([pt[1] for pt in corner_points])
                max_y = max([pt[1] for pt in corner_points])

                box_top_left = (min_x, min_y)
                box_bottom_right = (max_x, max_y)

                check_box_size(box_top_left, box_bottom_right)

                cv2.rectangle(image, box_top_left, box_bottom_right, (255, 0, 0), 2)

                if min_x <= left_pupil_x <= max_x and min_y <= left_pupil_y <= max_y:
                    mapped_x = map_range(left_pupil_x, min_x, max_x, 0, screen_width)
                    mapped_y = map_range(left_pupil_y, min_y, max_y, 0, screen_height)

                    try:
                        pyautogui.moveTo(int(mapped_x), int(mapped_y))
                    except pyautogui.FailSafeException:
                        pyautogui.moveTo(0, 0)

    if len(corner_points) < 4:
        cv2.putText(image, f'Look at {corner_labels[len(corner_points)]} corner and press "c"', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Eye Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
