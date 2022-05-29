# unet.py
# Source: https://github.com/hojonathanho/diffusion/blob/master/
# diffusion_tf/models/unet.py
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class TimestepEmbedding(layers.Layer):
	def __init__(self, dim, **kwargs):
		super().__init__()
		self.dim = dim


	def call(self, t):
		return get_timestep_embedding(t, self.dim)


class Upsample(layers.Layer):
	def __init__(self, with_conv=True, **kwargs):
		super().__init__()
		self.with_conv = with_conv
		self.conv = layers.Conv2D(filters=3, strides=1)


	def call(self, inputs):
		batch_size, height, width, channels = inputs.shape
		x = tf.image.resize(
			inputs, size=[height * 2, width * 2],
			method="nearest"
		)
		assert x.shape == [batch_size, height * 2, width * 2, channels]
		if self.with_conv:
			x = self.conv(x)
			assert x.shape == [batch_size, height * 2, width * 2, channels]
		return x


class Downsample(layers.Layer):
	def __init__(self, with_conv=True, **kwargs):
		super().__init__()
		self.with_conv = with_conv
		self.conv = layers.Conv2D(3, strides=2)
		self.avg_pool = layers.AveragePooling2D(
			strides=2, padding="same"
		)


	def call(self, inputs):
		batch_size, height, width, channels = inputs.shape
		if self.with_conv:
			x = self.conv(inputs)
		else:
			x = self.avg_pool(inputs)	
		assert x.shape == [batch_size, height // 2, width // 2, channels]
		return x


class ResNetBlock(layers.Layer):
	def __init__(self, out_ch=None, conv_shortcut=False, dropout, 
			**kwargs):
		super().__init__()
		self.out_ch = out_ch
		self.conv_shortcut = conv_shortcut
		self.dropout = dropout


	def build(self, input_shape):
		batch_size, height, width, channels = input_shape
		if self.out_ch is None:
			self.out_ch = channels
		self.c_not_out_ch = channels != self.out_ch

		# Layers.
		self.group_norm1 = tfa.layers.GroupNormalization()
		self.non_linear1 = layers.Activation("swish")
		self.conv1 = layers.Conv2D(self.out_ch)

		self.non_linear2 = layers.Activation("swish")
		self.dense2 = layers.Dense(self.out_ch)#layers.Conv2D(self.out_ch)

		self.group_norm3 = tfa.layers.GroupNormalization()
		self.non_linear3 = layers.Activation("swish")
		self.dropout3 = layers.Dropout(self.dropout)
		self.conv3 = layers.Conv2D(self.out_ch)

		self.conv4 = layers.Conv2D(self.out_ch)
		self.dense4 = layers.Dense(self.out_ch)


	def call(self, inputs, temb):
		x = inputs

		x = self.group_norm1(x)
		x = self.non_linear1(x)
		x = self.conv1(x)

		# Add in timestep embedding.
		x = self.non_linear2(temb)
		x += self.dense2(x)[:, None, None, :]

		x = self.group_norm3(x)
		x = self.non_linear3(x)
		x = self.dropout3(x)
		x = self.conv3(x)

		if self.c_not_out_ch:
			if self.conv_shortcut:
				inputs = self.conv4(inputs)
			else:
				inputs = self.dense4(inputs)

		assert x.shape == inputs.shape
		return inputs + x


class AttentionBlock(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__()


	def build(self, input_shape):
		batch_size, height, width, channels = input_shape

		# Layers.
		self.group_norm1 = tfa.layers.GroupNormalization()
		self.q_dense = layers.Dense(channels)
		self.k_dense = layers.Dense(channels)
		self.v_dense = layers.Dense(channels)

		self.softmax = layers.Softmax()

		self.out = layers.Dense(channels)


	def call(self, inputs):
		batch_size, height, width, channels = input_shape
		h = self.group_norm1(inputs)
		q = self.q_dense(h)
		k = self.k_dense(h)
		v = self.v_dense(h)

		w = tf.einsum('bhwc,bHWc->bhwHW', q, k) *\
			(int(channels) ** (-0.5))
		w = tf.reshape(w, [batch_size, height, width, channels * width])
		w = self.softmax(w)
		w = tf.reshape(w, [batch_size, height, width, height, width])

		h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
		h = self.out(h)
		return inputs + h


def create_unet_model(x, t, y, num_classes, ch, out_ch, 
		ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, 
		dropout=0.0, resample_with_conv=True):
	batch_size, s, _, _, = x.shape
	assert x.dtype == tf.float32 and x.shape[2] == s
	assert t.dtype in [int32, tf.int64]
	num_resolutions == len(ch_mult)
	
	assert num_classes == 1 and y is None, "not supported"
	del y

	# Timestep embedding.
	# temb = nn.get_timestep_embedding(t, ch)
	temb = TimestepEmbedding(ch)(t)
	temb = layers.Dense(ch * 4)(temb)
	temb = layers.Activation("swish")(temb)
	temb = layers.Dense(ch * 4)(temb)
	assert temb.shape == [batch_size, ch * 4]

	# Downsampling.
	hs = [layers.Conv2D(ch)(x)]
	for i_level in range(num_resolutions):
		# Residual blocks for this resolution.
		for i_block in range(num_res_blocks):
			h = ResNetBlock(
				out_ch=ch * ch_mult[i_level], dropout
			)(hs[-1], temb)
			if h.shape[1] in attn_resolutions:
				h = AttentionBlock()(h)
			hs.append(h)

		# Downsample.
		if i_level != num_resolutions - 1:
			hs.append(
				Downsample(with_conv=resample_with_conv)(hs[-1])
			)

	# Middle.
	h = hs[-1]
	h = ResNetBlock(dropout)(h, temb)
	h = AttentionBlock()(h)
	h = ResNetBlock(dropout)(h, temb)

	# Upsampling.
	for i_level in reversed(range(num_resolutions)):
		# Residual blocks for this resolution.
		for i_block in range(num_res_blocks + 1):
			h = ResNetBlock(
				out_ch=ch * ch_mult[i_level], dropout
			)(tf.concat([h, hs.pop()], axis=-1), temb)
			if h.shape[1] in attn_resolutions:
				h = AttentionBlock()(h)

		# Upsample.
		if i_level != 0:
			h = Upsample(with_conv=resample_with_conv)(h)
	assert not hs

	# End.
	h = layers.Activation("swish")(h)
	h = tfa.layers.GroupNormalization()(h)
	h = layers.Conv2D(out_ch)
	assert h.shape == x.shape[:3] + [out_ch]
	return keras.Model(
		inputs=[
			layers.Input(shape=x.shape), Layers.Input(shape=t.shape),
		], 
		outputs=[h], name="unet"
	)