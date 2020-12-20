import os
from flask import Flask, render_template, Response
import cv2
import uuid

app = Flask(__name__)
camera = cv2.VideoCapture(0)

MAX_COUNT = 200
files_folder = os.path.join(os.getcwd(), "files")


def record_frames():
    images_folder = os.path.join(files_folder, uuid.uuid4().hex)
    os.mkdir(images_folder)
    count = 0
    while True:
        success, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)), interpolation=cv2.INTER_AREA)
        print(frame.shape)
        if not success:
            break
        else:
            count += 1
            file_name = os.path.join(images_folder, "frame" + str(count) + ".tiff")
            cv2.imwrite(file_name, frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            if count >= MAX_COUNT:
                break


@app.route('/record_video')
def video_feed():
    return Response(record_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recording')
def recording():
    return render_template('recording.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
