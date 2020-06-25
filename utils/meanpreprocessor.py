# import
import cv2

class MeanPreprocessor:

	def __init__(self, rMean, gMean, bMean):
		self.rMean = rMean
		self.gMean = gMean
		self.bMean = bMean

	def preprocess(self, image):
		# split the images into channels
		(B, G, R) = cv2.split(image.astype("float32"))

		# subtract the means
		R -= self.rMean
		G -= self.gMean
		B -= self.bMean

		return cv2.merge([B, G, R])