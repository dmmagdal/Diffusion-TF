# test.py
# Simple program that iteratively tests out the core components of the
# altered implementations of the diffusion model.
# Reference (Custom Training Loop): https://www.tensorflow.org/guide/
# keras/writing_a_training_loop_from_scratch
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from gaussian_diffusion import GaussianDiffusion2, get_beta_schedule
from unet import create_unet_model
import datasets


def main():
	# Diffusion model hyperparameters.
	beta_start = 0.0009
	beta_end = 0.02
	num_diffusion_timesteps = 1000
	beta_schedule = "linear"

	model_mean_type = "eps"
	model_var_type = "fixedlarge"
	loss_type = "mse"

	# Instantiate a diffusion model.
	diffusion_model = GaussianDiffusion2(
		betas=get_beta_schedule(
			beta_schedule, beta_start=beta_start, beta_end=beta_end,
			num_diffusion_timesteps=num_diffusion_timesteps
		), model_mean_type=model_mean_type, 
		model_var_type=model_var_type, loss_type=loss_type
	)

	batch_size = 32

	# UNet hyperparameters.
	channels = 3
	dropout_rate = 0.1
	y = None
	num_classes = 1
	ch = 128 # run_cifar.py
	ch_mult = (1, 2, 2, 2)
	num_res_blocks = 2
	attn_resolutions = (16,)
	out_ch = (channels * 2) \
		if diffusion_model.model_var_type == "learned" else channels
	# Input image of shape (batch_size, height, width, channels).
	x = layers.Input(shape=(256, 256, 3), dtype=tf.float32)
	# Random normal of shape (batch_size,) with a mean 0 and stddev 
	# num_timesteps.
	t = tf.random.normal(
		[batch_size], 0, diffusion_model.num_timesteps, dtype=tf.float32#dtype=tf.int32
	)
	t = layers.Input(shape=(), dtype=tf.float32)#tf.int32)

	# Instantiate a UNet model.
	unet = create_unet_model(
		x=x, t=t, y=y, ch=ch, ch_mult=ch_mult, 
		num_res_blocks=num_res_blocks, num_classes=num_classes,
		attn_resolutions=attn_resolutions, out_ch=out_ch, 
		dropout=dropout_rate
	)
	unet.summary()

	# Load CIFAR10 dataset (cache in parent directory).
	data = tfds.load("cifar10")#, data_dir=".") # cache is finicky for some reason right now.
	print(data.keys())
	train_data = data["train"]
	test_data = data["test"]
	print(train_data)

	train_data = train_data.prefetch(tf.data.AUTOTUNE)\
		.shuffle(batch_size * 100)\
		.batch(
			batch_size, drop_remainder=True, 
			# num_parallel_calls=tf.data.AUTOTUNE
		)

	# Load CIFAR10 dataset using datasets.py
	# dataset = "cifar10"
	# tfds_data_dir = "./tensorflow_datasets"
	# ds = datasets.get_dataset(dataset, tfds_data_dir=tfds_data_dir)

	# Create a custom training model 
	lr = 2e-4
	epsilon = 1e-8
	model_opt = tf.keras.optimizers.Adam(
		learning_rate=lr, epsilon=epsilon
	)
	cifarDM = CifarDiffusionModel(diffusion_model, unet, (256, 256, 3))
	cifarDM.compile(optimizer=model_opt)
	cifarDM.fit(train_data, epochs=10, batch_size=batch_size)

	# Exit the program.
	exit(0)


class CifarDiffusionModel(tf.keras.Model):
	def __init__(self, diffusion, unet, img_shape):
		super().__init__()
		self.diffusion = diffusion
		self.unet = unet
		self.img_shape = img_shape


	def train_step(self, data):
		x = tf.cast(data["image"], dtype=tf.float32) # Convert image from tf.uint8 to tf.float32
		if x.shape[1:].as_list() != self.img_shape:
			x = tf.image.resize(x, self.img_shape[:2], "nearest")

		with tf.GradientTape() as tape:
			loss = train_fn(x, None, self.unet, self.diffusion)

		grads = tape.gradient(loss["loss"], self.unet.trainable_weights)
		self.optimizer.apply_gradients(
			zip(grads, self.unet.trainable_weights)
		)

		return loss


	def sample(self, num_samples):
		noise = tf.randome.uniform(
			[num_samples], 0, self.diffusion.num_timesteps
		)
		return samples_fn(noise, self.unet, self.diffusion)


def train_fn(x, y, unet, diffusion, randflip=True):
	batch_size, height, width, channels = x.shape
	if randflip:
		x = tf.image.random_flip_left_right(x)
		assert x.shape.as_list() == [batch_size, height, width, channels]

	t = tf.random.uniform(
		[batch_size], 0, diffusion.num_timesteps, dtype=tf.int32
	)
	losses = diffusion.training_losses(
		denoise_fn=unet, x_start=x, t=t
	)
	assert losses.shape.as_list() == t.shape.as_list() == [batch_size]
	return {"loss": tf.reduce_mean(losses)}


def samples_fn(dummy_noise, unet, diffusion):
	return{
		"samples": diffusion.p_sample_loop(
			denoise_fn=unet, shape=dummy_noise.shape.as_list(),
			noise_fn=tf.random.normal
		),
	}


def progressive_samples_fn(dummy_noise, unet, diffusion):
	samples, progressive_samples = diffusion.p_sample_loop_progressive(
		denoise_fn=unet, shape=dummy_noise.shape.as_list(),
		noise_fn=tf.random.normal
	)
	return {
		"samples": samples,
		"progressive_samples": progressive_samples,
	}


def bpd_fn(x, y, unet, diffusion):
	total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(
		denoise_fn=unet, x_start=x
	)
	return {
		"total_bpd": total_bpd_b,
		"terms_bpd": terms_bpd_bt,
		"prior_bpd": prior_bpd_b,
		"mse": mse_bt,
	}


if __name__ == '__main__':
	main()