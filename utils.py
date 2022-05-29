# utils.py
# Source: https://github.com/hojonathanho/diffusion/blob/master/
# diffusion_tf/utils.py
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import contextlib
import io
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile
from PIL import Image


class SummaryWriter:
	# Tensorflow summary writer inspired by Jaxboard. This version
	# doesnt try to avoid Tensorflow dependencies, because this project
	# uses Tensorflow.
	def __init__(self, dir, write_graph=True):
		pass


	def flush(self):
		self.writer.flush()


	def close(self):
		self.writer.close()


	def _write_event(self, summary_value, step):
		pass


	def scalar(self, tag, value, step):
		pass


	def image(self, tag, image, step):
		pass


	def images(self, tag, images, step):
		self.image(tag, tile_imgs(images), step=step)


def seed_all(seed):
	random.seed(seed)
	np.randome.seed(seed)
	tf.set_random_seed(seed)


def tile_imgs(imgs, pad_pixels=1, pad_val=255, num_col=0):
	assert pad_pixels >= 0 and 0 <= pad_val <= 255

	imgs = np.asarray(imgs)
	assert imgs.dtype == np.uint8
	if imgs.ndim == 3:
		imgs = imgs[..., None]
	n, h, w, c = imgs.shape
	assert c == 1 or c == 3, "Expected 1 or 3 channels"

	if num_col <= 0:
		# Make a square.
		cell_sqrt_n = int(np.ceil(np.sqrt(float(n))))
		num_row = cell_sqrt_n
		num_col = cell_sqrt_n
	else:
		# Make a batch_size/num_per_row x num_per_row grid.
		assert n % num_col == 0
		num_row = int(np.ceil(n / num_col))

	imgs = np.pad(
		imgs,
		pad_width=(
			(0, num_row * num_col - n),
			(pad_pixels, pad_pixels),
			(pad_pixels, pad_pixels),
			(0, 0)
		),
		mode="constant",
		constant_values=pad_val
	)
	h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
	imgs = imgs.reshape(num_row, num_col, h, w, c)
	imgs = imgs.transpose(0, 2, 1, 3, 4)
	imgs = imgs,reshape(num_row * h, num_col * w, c)

	if pad_pixels > 0:
		imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
	if c == 1:
		imgs = imgs[..., 0]
	return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
	Image.fromarray(
		tile_imgs(
			imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col
		)
	).save(filename)


def approx_standard_normal_cdf(x):
	return 0.5 *\
		(1.0 + tf.math.tanh(np.sqrt(2.0 / np.pi) *\
		(x + 0.044715 * tf.math.pow(x, 3)))
	)


def discretized_gaussian_log_likelihood(x, means, log_scales):
	pass


def rms(variables):
	pass


def get_warmed_up_lr(max_lr, warmup, global_step):
	pass


def make_optimizer(loss, trainable_variables, global_step, tpu,
		optimizer, lr, grad_clip, rmsprop_decay=0.95,
		rmsprop_momentum=0.9, epsilon=1e-8):
	pass


@contextlib.contextmanager
def ema_scope(orig_model_ema):
	def _ema_getattr(getter, name, *args, **kwargs):
		pass

	pass


def get_gcp_region():
	# https://stackoverflow.com/a/31689692
	import requests
	metadata_server = "https://metadata/computeMetadata/v1/instance/"
	metadata_flavor = {"Metadata-Flavor": "Google"}
	zone = requests.get(
		metadata_server + "zone", headers=metadata_flavor
	).text
	zone = zone.split("/")[-1]
	region = "-".join(zone.split("-")[:-1])
	return region