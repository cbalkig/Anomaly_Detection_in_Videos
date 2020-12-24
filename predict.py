import os
import numpy as np
from keras.models import load_model


class Config:
    MODEL_PATH = os.path.join("model", "model.hdf5")
    IMAGE_SIZE = 256


def get_test_data(img):
    img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img = np.array(img, dtype=np.float32) / 256.0
    img = np.reshape(img, img.shape + (1,))
    return img


def get_model():
    return load_model(Config.MODEL_PATH)


def evaluate(model, img):
    frame = get_test_data(img)
    frame = np.array(frame)
    frame = np.reshape(frame, (1,) + frame.shape)
    prediction = model.predict(frame)
    return str(np.max(prediction) * 100)
