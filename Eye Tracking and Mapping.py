import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]

screen_width, screen_height = pyautogui.size()

x_min = 287
x_max = 391
y_min = 170
y_max = 209

pyautogui.FAILSAFE = False

smoothing_factor = 0.3

# Initialize previous screen coordinates for smoothing
prev_x_screen, prev_y_screen = None, None

def map_iris_to_screen(x_iris, y_iris):
    x_screen = int((x_iris - x_min) / (x_max - x_min) * screen_width)
    y_screen = int((y_iris - y_min) / (y_max - y_min) * screen_height)
    return x_screen, y_screen

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, frame = cap.read()
    
    if not success:
        print("Ignoring empty camera frame")
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            left_iris_x = sum([face_landmarks.landmark[i].x for i in LEFT_IRIS_LANDMARKS]) / len(LEFT_IRIS_LANDMARKS)
            left_iris_y = sum([face_landmarks.landmark[i].y for i in LEFT_IRIS_LANDMARKS]) / len(LEFT_IRIS_LANDMARKS)
            
            right_iris_x = sum([face_landmarks.landmark[i].x for i in RIGHT_IRIS_LANDMARKS]) / len(RIGHT_IRIS_LANDMARKS)
            right_iris_y = sum([face_landmarks.landmark[i].y for i in RIGHT_IRIS_LANDMARKS]) / len(RIGHT_IRIS_LANDMARKS)

            avg_iris_x = (left_iris_x + right_iris_x) / 2
            avg_iris_y = (left_iris_y + right_iris_y) / 2

            avg_iris_x_pixel = int(avg_iris_x * frame.shape[1])
            avg_iris_y_pixel = int(avg_iris_y * frame.shape[0])

            x_screen, y_screen = map_iris_to_screen(avg_iris_x_pixel, avg_iris_y_pixel)
            
            # Smoothing the movement with exponential moving average (EMA)
            if prev_x_screen is not None and prev_y_screen is not None:
                x_screen = int(smoothing_factor * x_screen + (1 + smoothing_factor) * prev_x_screen)
                y_screen = int(smoothing_factor * y_screen + (1 + smoothing_factor) * prev_y_screen)

            prev_x_screen, prev_y_screen = x_screen, y_screen

            try:
                pyautogui.moveTo(x_screen, y_screen)
            except pyautogui.FailSafeException:
                screen_width, screen_height = pyautogui.size()
                pyautogui.moveTo(0, 0)
                print("FailSafe triggered! Moving cursor back to the center.")

            cv2.circle(frame, (avg_iris_x_pixel, avg_iris_y_pixel), 5, (255, 0, 0), -1)

    cv2.imshow('Iris Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
