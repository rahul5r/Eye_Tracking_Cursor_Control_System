import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True)
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

eye_landmarks = {
    'Top Left Eyelid': 159, 'Bottom Left Eyelid': 145,
    'Top Right Eyelid': 386, 'Bottom Right Eyelid': 374,
}

def check_blink(landmarks):
    left_eye = abs(landmarks[0][1] - landmarks[1][1])
    right_eye = abs(landmarks[2][1] - landmarks[3][1])
    if left_eye < 4 and right_eye < 4:
        return 'Both'
    elif left_eye < 4:
        return 'Left'
    elif right_eye < 4:
        return 'Right'
    else:
        return None

while True:
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            
            #if cv2.waitKey(1) == ord('c'):
                ih, iw, ic = img.shape
                landmarks = []
                
                for eye_part, indices in eye_landmarks.items():
                    lm = faceLms.landmark[indices]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks.append([x,y])
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                    # print(f"{eye_part} - Eyelid landmark {indices}: ({x}, {y})")
                blink = check_blink(landmarks)
                if blink:
                    cv2.putText(img, f"{blink} Eye Blink", (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
