# run_cifar.py
# Source: https://github.com/hojonathanho/diffusion/blob/master/
# scripts/run_cifar.py
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf
import utils
from gaussian_diffusion2 import get_beta_schedule, GaussianDiffusion2
from unet import create_unet_model
from tpu_utils import tpu_utils, datasets, simple_eval_worker


class Model(tpu_utils.Model):
	def __init__(self, model_name, betas, model_mean_type, model_var_type, loss_type, num_classes, dropout, randflip):
		self.model_name = model_name
		self.diffusion = GaussianDiffusion2(
			betas=betas, model_mean_type=model_mean_type,
			model_var_type=model_var_type, loss_type=loss_type,
		)
		self.num_classes = num_classes
		self.dropout = dropout
		self.randflip = randflip


	def _denoise(self, x, t, y, dropout):
		batch_size, height, width, channels = x.shape.as_list()
		assert x.dtype == tf.float32
		assert t.shape == [batch_size] and t.dtype in [tf.int32, tf.int64]
		assert y.shape == [batch_size] and y.dtype in [tf.int32, tf.int64]
		out_ch = (channels * 2) if self.diffusion.model_var_type == "learned" else channels
		y = None
		if self.model_name == "unet2d16b2": # 35.7M
			return create_unet_model(
				x, t=t, y=y, ch=128, ch_mult=(1, 2, 2, 2),
				num_res_blocks=2, attn_resolutions=(16,), 
				out_ch=out_ch, num_classes=self.num_classes,
				dropout=dropout
			)
		raise NotImplementedError(self.model_name)


	def train_fn(self, x, y):
		batch_size, height, width, channels = x.shape.as_list()
		if self.randflip:
			x = tf.image.random_flip_left_right(x)
			assert x.shape == [batch_size, height, width, channels]
		t = tf.random.normal(
			[batch_size], 0, self.diffusion.num_timesteps, dtype=tf.int32
		)
		losses = self.diffusion.training_losses(
			denoise_fn=functools.partial(
				self._denoise, y=y, dropout=self.dropout
			),
			x_start=x, 
			t=t,
		)
		assert shape == t.shape == [batch_size]
		return {"loss": tf.reduce_mean(losses)}


	def samples_fn(self, dummy_noise, y):
		return {
			"samples": self.diffusion.p_sample_loop(
				denoise_fn=functools.partial(
					self._denoise, y=y, dropout=0
				),
				shape=dummy_noise.shape.as_list(),
				noise_fn=tf.random.normal,
			)
		}


	def progressive_samples_fn(self, dummy_noise, y):
	samples, progressive_samples = self.diffusion.p_sample_loop_progressive(
		denoise_fn=functools.partial(
			self._denoise, y=y, dropout=0
		),
		shape=dummy_noise.shape.as_list(),
		noise_fn=tf.random.normal,
	)
	return {
		"samples": samples, "progressive_samples": progressive_samples
	}


	def bpd_fn(self, x, y):
		total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
			denoise_fn=functools.partial(
				self._denoise, 
				y=y, 
				dropout=0
			),
			x_start=x,
		)
		return {
			"total_bpd": total_bpd_b,
			"terms_bpd": terms_bpd_bt,
			"prior_bpd": prior_bpd_b,
			"mse": mse_bt,
		}


def _load_model(kwargs, ds):
	return Model(
		model_name=kwargs["model_name"],
		betas=get_beta_schedule(
			kwargs["beta_schedule"], beta_start=kwargs["beta_start"],
			beta_end=kwargs["beta_end"],
			num_diffusion_timesteps=kwargs["num_diffusion_timesteps"],
		),
		model_mean_type=kwargs["model_mean_type"],
		model_var_type=kwargs["num_diffusion_timesteps"],
		loss_type=kwargs["loss_type"],
		num_classes=ds.num_classes,
		dropout=kwargs["dropout"],
		randflip=kwargs["randflip"],
	)


def simple_eval(model_dir, tpu_name, bucket_name_prefix, mode, load_ckpt, 
		total_bs=256, tfds_data_dir="tensorflow_datasets"):
	region = utils.get_gcp_region()
	tfds_data_dir = 'gs://{}-{}/{}'.format(
		bucket_name_prefix, region, tfds_data_dir
	)
	kwargs = tpu_utils.load_train_kwargs(model_dir)
	print("loaded kwargs", kwargs)
	ds = datasets.get_dataset(
		kwargs["datasets"], tfds_data_dir=tfds_data_dir
	)
	worker = simple_eval_worker.SimpleEvalWorker(
		tpu_name=tpu_name, 
		model_constructor=functools.partial(
			_load_model, kwargs=kwargs, ds=ds
		),
		total_bs=total_bs,
		datasets=ds
	)
	worker.run(mode=mode, logdir=model_dir, load_ckpt=load_ckpt)


def evaluation():
	pass


def train(exp_name, tpu_name, bucket_name_prefix, 
		model_name="unet2d16b2", dataset="cifar10", optimizer="adam",
		total_bs=128, grad_clip=1, lr=2e-4, warmup=5000,
		num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02,
		beta_schedule="linear", model_mean_type="eps", 
		model_var_type="fixedlarge", loss_type="mse", dropout=0.1,
		randflip=1, tfds_data_dir="tensorflow_datasets", 
		log_dir="logs", keep_checkpoint_max=5):
	region = utils.get_gcp_region()
	tfds_data_dir = "gs:{}-{}/{}".format(
		bucket_name_prefix, region, tfds_data_dir
	)
	log_dir = "gs:[]-{}/{}".format(
		bucket_name_prefix, region, log_dir
	)
	ds = datasets.get_dataset(dataset, tfds_data_dir=tfds_data_dir)
	tpu_utils.run_training(
		date_str="9999-99-99",
		model_constructor=lambda: Model(
			model_name=model_name,
			betas=get_beta_schedule(
				beta_schedule, beta_start=beta_start, beta_end=beta_end,
				num_diffusion_timesteps=num_diffusion_timesteps
			),
			model_mean_type=model_mean_type,
			model_var_type=model_var_type,
			loss_type=loss_type, num_classes=num_classes,
			dropout=dropout, randflip=randflip
		),
		optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup,
		grad_clip=grad_clip, train_input_fn=ds.train_input_fn,
		tpu=tpu_name, log_dir=log_dir, iterations_per_loop=2000,
		keep_checkpoint_max=keep_checkpoint_max
	)