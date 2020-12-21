import os
from flask import Flask, render_template, Response, request
from PIL import Image

app = Flask(__name__)
files_folder = os.path.join(os.getcwd(), "files")


@app.route('/')
def index():
    return render_template('index.html')


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/recording')
def recording():
    return Response(open('./static/recording.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


@app.route('/test')
def test():
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

    image = Image.open(TEST_IMAGE_PATHS[0])
    objects = None #object_detection_api.get_objects(image)

    return objects


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