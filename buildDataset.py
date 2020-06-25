# import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import config
from utils.aspectawarepreprocessor import AspectAwarePreprocessor
from utils.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[2].split(".")[0] 
	for p in trainPaths]

# encode the labels
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# split the dataset into training, validation and testing
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(
	trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
	stratify=trainLabels)

(trainPaths, valPaths, trainLabels, valLabels) = train_test_split(
	trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES,
	stratify=trainLabels)

# construct dataset attributes
dataset = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5),
]

# initialize the image preprocessor and list for RGB mean values
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in dataset:
	print("[INFO] building {}...".format(outputPath))

	# create HDF5 writer
	writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

	# loop over the images
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image and preprocess it
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# if training dataset, store mean values of RGB channel
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# add image and label to HDF5 dataset
		writer.add([image], [label])

	# close the HDF5 writer
	writer.close()

# construct dictionary of mean RGB values and serialize to JSON file
print("[INFO] serealizing means...")
D = {"R": np.mean(R),
	"G": np.mean(G),
	"B": np.mean(B)
}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
