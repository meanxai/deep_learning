# [MXDL-12-02] 1D Convolution
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/NHY6y3UWvwQ
#
import tensorflow as tf

class MyConv1D(tf.keras.layers.Layer):
	def __init__(self, input_dims, filters, kernel_size, padding='VALID'):
		super().__init__()
		self.k_size = kernel_size
		self.pad = padding
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(w_init([filters, kernel_size, input_dims]))
		self.b = tf.Variable(tf.zeros_initializer()([filters, ]))
		
	def call(self, x):
		n_rows = x.shape[1]
		if self.pad == 'SAME':
			n_pad_top = n_pad_bot = self.k_size // 2
			px = tf.pad(x, [[0,0], [n_pad_top, n_pad_bot], [0,0]])
			n_outs = n_rows    # the number of rows in the feature map
		else: # no padding
			px = x
			n_outs = n_rows - self.k_size + 1

             # Compute the cross-correlations as we move down the filters.
		cc = []
		for k in range(n_outs):
			p = px[:, k:(k + self.k_size), :]
			cc.append(self.w[tf.newaxis, :, :, :] * p[:, tf.newaxis, :, :])

		cc = tf.stack(cc)   # [n_out, None, filters, kernel_size, input_dims]
		conv = tf.reduce_sum(cc, [3, 4])            # [n_outs, None, filters]
		conv = tf.transpose(conv, [1, 0, 2])        # [None, n_outs, filters]
		conv += self.b[tf.newaxis, tf.newaxis, :]   # [None, None, filters]
		return conv                                 # [None, n_outs, filters]
    
