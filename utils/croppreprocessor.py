# import
import numpy as np
import cv2

class CropPreprocessor:

	def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.horiz = horiz
		self.inter = inter

	def preprocess(self, image):
		# initialize list of crops

		# grab height and width of image
		(h, w) = image.shape[:2]

		# define the co-ordinates of the corners of the crop
		coords = [
			[0, 0, self.width, self.height],
			[w - self.width, 0, w, self.height],
			[w - self.width, h - self.height, w, h],
			[0, h - self.height, self.width, h]
		]

		# compute center crop
		dW = int(0.5 * (w - self.width))
		dH = int(0.5 * (h - self.height))
		coords.append([dW, dH, w - dW, h - dH])

		# loop over the coords, extract the crop and resize them
		# to a fized scale
		for (startX, startY, endX, endY) in coords:
			crop = image[startY:endY, startX:endX]
			crop = cv2.resize(crop, (self.width, self.height),
				interpolation=self.inter)
			crops.append(crop)

		# perform horizontal flips if required
		if self.horiz:
			mirrors = [cv2.flip(c, 1) for c in crops]
			crops.extend(mirrors)

		return np.array(crops)