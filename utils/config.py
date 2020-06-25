# import 
import os

# path to original dataset
DIRECTORY = "../dataset/kaggle_dogs_vs_cats"
IMAGES_PATH = os.path.sep.join([DIRECTORY, "train"])

# number of images for validation and testing
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# path to hdf5 files to be stored
TRAIN_HDF5 = os.path.sep.join([DIRECTORY, "hdf5", "train.hdf5"])
VAL_HDF5 = os.path.sep.join([DIRECTORY, "hdf5", "val.hdf5"])
TEST_HDF5 = os.path.sep.join([DIRECTORY, "hdf5", "test.hdf5"])

# path to output files
OUTPUT_PATH = "output"

MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "model.h5"]) 
DATASET_MEAN = os.path.sep.join([OUTPUT_PATH, "dataset_mean.json"]) 
