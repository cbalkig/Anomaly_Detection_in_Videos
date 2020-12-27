import os

working_directory = '.'


class Config:
    DATASET_PATH = os.path.join(working_directory, "files")
    EDIT_DATASET_PATH = os.path.join(working_directory, "regenerated_files")
    BATCH_SIZE = 128
    EPOCHS = 10
    MODEL_PATH = os.path.join(working_directory, "model.hdf5")
    IMAGE_SIZE = 256
    NUM_THREADS = 50
