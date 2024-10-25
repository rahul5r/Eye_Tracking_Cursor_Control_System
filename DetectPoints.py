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
    'Center' : 168,
    'Left Iris': [468, 469, 470, 471, 472],
    'Right Iris': [473, 474, 475, 476, 477]
}

while True:
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            ih, iw, ic = img.shape
            
            for eye_part, indices in eye_landmarks.items():
                if isinstance(indices, list):
                    for iris_index in indices:
                        lm = faceLms.landmark[iris_index]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                        print(f"{eye_part} - Iris landmark {iris_index}: ({x}, {y})")
                
                elif eye_part == "Center":
                    lm = faceLms.landmark[indices]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                    print(f"{eye_part} - landmark {indices}: ({x}, {y})")
                else:
                    lm = faceLms.landmark[indices]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                    print(f"{eye_part} - Eyelid landmark {indices}: ({x}, {y})")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
