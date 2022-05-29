# fid.py
# Calculate/implement Frechet Inception Distance (FID) score.
# Source: https://machinelearningmastery.com/how-to-implement-the-
# frechet-inception-distance-fid-from-scratch/
# Source (tensorflow Inception v3): https://www.tensorflow.org/
# api_docs/python/tf/keras/applications/inception_v3
# Tensorfow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from numpy.random import random
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from scipy.linalg import sqrtm
from skimage.transform import resize


def calculate_fid(act1, act2):
	# Calculate the mean and covariance statistics.
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

	# Calculate the sum squared difference between means.
	ssdiff = np.sum((mu1 - mu2) ** 2.0)
	
	# Calculate the sqrt of product between cov.
	covmean = sqrtm(sigma1.dot(sigma2))

	# Check and correct imaginary numbers from sqrt.
	if iscomplexobj(covmean):
		covmean = covmean.real

	# Calculate score.
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def calculate_fid_keras(model, img1, img2):
	# Calculate the activations.
	act1 = model.predict(img1)
	act2 = model.predict(img2)

	# Calculate the mean and covariance statistics.
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

	# Calculate the sum squared difference between means.
	ssdiff = np.sum((mu1 - mu2) ** 2.0)
	
	# Calculate the sqrt of product between cov.
	covmean = sqrtm(sigma1.dot(sigma2))

	# Check and correct imaginary numbers from sqrt.
	if iscomplexobj(covmean):
		covmean = covmean.real

	# Calculate score.
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# Resize with nearest neighbor interpolation.
		new_image = resize(image, new_shape, 0)

		# Store.
		images_list.append(new_image)
	return asarray(images_list)


def main():
	# Implement FID in Numpy. Create two collections of activations
	# (random).
	act1 = random(10 * 1024)
	act1 = act1.reshape((10, 1024))
	act2 = random(10 * 1024)
	act2 = act2.reshape((10, 1024))

	# FID between act1 and act1 (expected to be 0.0 or -0.0).
	print("FID (same): %.3f" % calculate_fid(act1, act1))

	# FID between act1 and act2 (expected to be some non-zero, positive
	# float).
	print("FID (different): %.3f" % calculate_fid(act1, act2))

	# Implement FID in Keras. Load (pretrained) Inception V3 model.
	# Remove the output (top) of the model via include_top=False. This
	# removes the global average pooling layer which we require, but
	# that can be added back via specifying pooling=avg. When the
	# output layer is removed, we must also specify the shape of the
	# input images.
	model = InceptionV3(
		include_top=False, pooling="avg", input_shape=(299, 299, 3)
	)
	model.summary()

	img1 = np.random.randint(0, 255, 10 * 32 * 32 * 3)
	img1 = img1.reshape((10, 32, 32, 3))
	img2 = np.random.randint(0, 255, 10 * 32 * 32 * 3)
	img2 = img2.reshape((10, 32, 32, 3))

	# Convert integer to float values.
	img1 = img1.astype("float32")
	img2 = img2.astype("float32")

	# Resize/scale images.
	img1 = scale_images(img1, (299, 299, 3))
	img2 = scale_images(img2, (299, 299, 3))

	# Scale the pixel values to meet the expectations of Inception v3
	# model.
	img1 = preprocess_input(img1)
	img2 = preprocess_input(img2)

	# FID between act1 and act1 (expected to be 0.0 or -0.0).
	print("FID (same): %.3f" % calculate_fid_keras(model, img1, img1))

	# FID between act1 and act2 (expected to be some non-zero, positive
	# float).
	print("FID (different): %.3f" % calculate_fid_keras(model, img1, img2))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()