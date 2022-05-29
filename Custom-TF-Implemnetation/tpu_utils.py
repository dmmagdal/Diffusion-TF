# tpu_utils.py


import json
import os
import pickle
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import trange
import utils


def run_inception(images):
	assert images.dtype == tf.float32 # images should be in [-1, 1]
	


class TPUModel:
	# All images (inputs and outputs) should be normalized to [-1, 1].
	def train_fn(self, x, y) -> dict:
		raise NotImplementedError


	def samples_fn(self, dummy_x, y) -> dict:
		raise NotImplementedError


	def sample_and_run_inception(self, dummy_x, y, clip_samples=True):
		samples_dict = self.samples_fn(dummy_x, y)
		assert isinstance(samples_dict, dict)
		return {
			k: run_inception(
				tf.clip_by_value(v, -1.0, 1.0) if clip_samples else v
			)
			for (k, v) in samples_dict.items()
		}


def run_training(model_constructor, train_input_fn, total_bs, 
		optimizer, lr, warmup, grad_clip, ema_decay=0.9999, tpu=None,
		zone=None, project=None, log_dir, exp_name, dump_kwargs=None,
		date_str=None, iterations_per_loop=1000, keep_checkpoint_max=2,
		max_steps=int(1e10)):
	# Create checkpoint directory.

	# Save kwargs to json format.


	# model_fn for TPUEstimator.
	def model_fn(features, params, mode):
		local_bs = params["batch_size"]
		print(
			"Global batch size: {}, local batch size: {}".format(
				total_bs, local_bs
			)
		)
		# assert total_bs == num_tpu_replicas() * local_bs

		assert mode = tf.estimator.ModeKeys.TRAIN, "only TRAIN mode supported"
		assert features["image"].shape[0] == local_bs
		assert features["label"].shape == [local_bs] and features["label"].dtype == tf.int32

		del params

		# Create model.
		model = model_controller()
		assert isinstance(mode, Model)

		# Training loss.
		train_info_dict = model.train_fn(
			normalize_data(tf.cast(features["image"], tf.float32)),
			features["label"]
		)
		loss = train_info_dict["loss"]
		assert loss..shape == []

		# Train op.
		trainable_variables = tf.compat.v1.trainable_variables()
		print("num params: {:,}".format(
			sum(int(np.prod(p.shape.as_list()))
			for p in trainable_variables)
		))
		global_step = tf.compat.v1.train.get_or_create_global_step()
		warmed_up_lr = utils.get_warmuped_up_lr(
			max_lr=lr, warmup=warmup, global_step=global_step
		)
		train_op, gnorm = utils.make_optimizer(
			loss=loss, trainable_variables=trainable_variables,
			global_step=global_step, lr=get_warmed_up_lr,
			optimizer=optimizer, 
			grad_clip=grad_clip / float(num_tpu_replicas()),
			tpu=True
		)

		# EMA.
		ema, ema_op = make_ema(
			global_step=global_step, ema_decay=ema_decay,
			trainable_variables=trainable_variables
		)
		with tf.compat.v1.control_dependencies([train_op]):
			train_op = tf.compat.v1.group(ema_op)

		# Summary.


	# Set up estimator and train.


class InceptionFeatures:
	# Compute and store Inception features for a dataset.
	def __init__(self, dataset, strategy, limit_dataset_size=0):
		# Distributed dataset iterator.

		# Inception network on the dataset.

		self.cached_inception_real = None # cached inception features
		self.real_incetion_score = None # saved inception scores for the dataset


	def get(self, sess):
		# On the first invocation, compute Inception activations for
		# the eval dataset.