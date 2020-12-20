import os

working_directory = '/Users/balki/files/'


class Config:
    DATASET_PATH = os.path.join(working_directory, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Train")
    SINGLE_TEST_PATH = os.path.join(working_directory, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Test", "Test032")
    BATCH_SIZE = 4
    EPOCHS = 3
    MODEL_PATH = os.path.join(working_directory, "model.hdf5")
