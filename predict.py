import os
from os import listdir
from os.path import join, isdir
import numpy as np
from PIL import Image
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
import matplotlib.pyplot as plt

working_directory = '/Users/balki/files/'
test_folder = "Test014"


class Config:
    DATASET_PATH = os.path.join(working_directory, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Train")
    SINGLE_TEST_PATH = os.path.join(working_directory, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Test", test_folder)
    BATCH_SIZE = 16
    FRAME_COUNT = 200
    MODEL_PATH = os.path.join(working_directory, "model.hdf5")


def get_model():
    return load_model(Config.MODEL_PATH, custom_objects={'LayerNormalization': LayerNormalization})


def get_single_test():
    sz = Config.FRAME_COUNT
    test = np.zeros(shape=(Config.FRAME_COUNT, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "tif":
            img = Image.open(join(Config.SINGLE_TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test


def evaluate(model):
    print("got model")
    test = get_single_test()
    print("got test")
    sz = test.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences, batch_size=Config.BATCH_SIZE)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0, sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.savefig(test_folder + '.png')
    plt.show()


if __name__ == '__main__':
    model = get_model()
    evaluate(model)
