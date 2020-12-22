import os
from flask import Flask, Response, request
from PIL import Image
from predict import get_model, evaluate

app = Flask(__name__)
files_folder = os.path.join(os.getcwd(), "files")
model = get_model()

images = {}

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    return Response(open('static/index.html').read(), mimetype="text/html")


@app.route('/anomalyDetection', methods=['POST'])
def anomaly_detection():
    image_file = request.files['image']
    uuid = request.form.get('uuid')
    image_object = Image.open(image_file).convert('L')
    if uuid not in images:
        images[uuid] = []
    images[uuid].append(image_object)
    if len(images[uuid]) is 10:
        evaluate(model, image_object)


@app.route('/recordingImage', methods=['POST'])
def image():
    image_file = request.files['image']
    uuid = request.form.get('uuid')
    count = request.form.get('count')
    image_object = Image.open(image_file).convert('L')
    images_folder = os.path.join(files_folder, uuid)
    if not os.path.exists(images_folder):
        try:
            os.mkdir(images_folder)
        except:
            print("Ignore")

    file_name = os.path.join(images_folder, str(count) + ".tif")
    image_object.save(file_name)
    return "save"


if __name__ == '__main__':
    #app.run(host="192.168.0.17", port=5000, debug=True)
    app.run(host="192.168.0.17", port=5000, debug=True, ssl_context=('ssl/server.crt', 'ssl/server.key'))