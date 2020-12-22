import os
import numpy as np
from keras.models import load_model
from keras_layer_normalization import LayerNormalization

working_directory = 'model'


class Config:
    BATCH_SIZE = 1
    FRAME_SIZE = 10
    MODEL_PATH = os.path.join(working_directory, "model.hdf5")


def get_model():
    return load_model(Config.MODEL_PATH, custom_objects={'LayerNormalization': LayerNormalization})


def get_single_test(images):
    test = np.zeros(shape=(Config.FRAME_SIZE, 256, 256, 1))
    cnt = 0
    for img in images:
        test[cnt, :, :, 0] = img
        cnt = cnt + 1
    return test


def evaluate(model, images):
    test = get_single_test(images)
    sequences = np.zeros((1, Config.FRAME_SIZE, 256, 256, 1))
    sequences[0] = test

    reconstructed_sequences = model.predict(sequences, batch_size=Config.BATCH_SIZE)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0, 1)])

    print(sequences_reconstruction_cost)
    return str(sequences_reconstruction_cost)

