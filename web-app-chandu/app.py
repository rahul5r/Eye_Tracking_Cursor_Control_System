from flask import Flask, render_template, jsonify
from face_recognition_model import start_face_recognition
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start-face-recognition")
def face_recognition():
    success = start_face_recognition()
    if success:
        try:
            process = subprocess.Popen(
                ["python", "Eye_Tracking_and_Clicking.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=5)  # Wait for a short duration
            if stderr:
                return jsonify({"success": False, "error": stderr.decode()})
            return jsonify({"success": True, "message": "Eye tracking started successfully."})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({"success": False, "message": "Face recognition failed."})

if __name__ == "__main__":
    app.run(debug=True)