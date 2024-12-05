from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)
tracking_active = False


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    global tracking_active
    while tracking_active:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    global tracking_active
    if tracking_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Tracking not active", 200


@app.route('/start', methods=['POST'])
def start_tracking():
    global tracking_active
    tracking_active = True
    return "Tracking started", 200


@app.route('/stop', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = False    
    camera.release()
    return "Tracking stopped", 200

if __name__ == '__main__':
    app.run(debug=True)
