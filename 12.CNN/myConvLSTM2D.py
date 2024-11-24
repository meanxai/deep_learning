# [MXDL-12-06] Convolutional LSTM
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/FzUAPtDgA_o
#
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv3D

class MyConvLSTM2D(Layer):
   def __init__(self, filters, kernel_size, pad, d):  # d: input dims
      super().__init__()
      self.nh = filters

      # Convolutional layers
      self.Wf = Conv2D(filters, kernel_size, padding=pad)  # forget
      self.Wi = Conv2D(filters, kernel_size, padding=pad)  # input
      self.Wc = Conv2D(filters, kernel_size, padding=pad)  # candidate
      self.Wo = Conv2D(filters, kernel_size, padding=pad)  # output
      self.Rf = Conv2D(filters, kernel_size, padding=pad)  # forget
      self.Ri = Conv2D(filters, kernel_size, padding=pad)  # input
      self.Rc = Conv2D(filters, kernel_size, padding=pad)  # candidate
      self.Ro = Conv2D(filters, kernel_size, padding=pad)  # output

      # Peephole connections
      w_init = tf.random_normal_initializer()
      self.Pf = tf.Variable(w_init([1,d[2],d[3],filters]),trainable=True)
      self.Pi = tf.Variable(w_init([1,d[2],d[3],filters]),trainable=True)
      self.Po = tf.Variable(w_init([1,d[2],d[3],filters]),trainable=True)

   def lstm_cell(self, x, h, c):
      f_gate = tf.math.sigmoid(self.Wf(x) + self.Rf(h) + self.Pf * c)
      i_gate = tf.math.sigmoid(self.Wi(x) + self.Ri(h) + self.Pi * c)
      c_gate = tf.math.tanh(self.Wc(x) + self.Rc(h))
      o_gate = tf.math.sigmoid(self.Wo(x) + self.Ro(h) + self.Po * c)
      c_stat = c * f_gate + c_gate * i_gate
      h_stat = tf.math.tanh(c_stat) * o_gate
      return h_stat, c_stat

   def call(self, x):
      h=tf.zeros(shape=(tf.shape(x)[0], x.shape[2], x.shape[3], self.nh))
      c=tf.zeros(shape=(tf.shape(x)[0], x.shape[2], x.shape[3], self.nh))

      # Repeat lstm_cell for the number of time steps
      for t in range(x.shape[1]):
         h, c = self.lstm_cell(x[:, t, :, :, :], h, c)
      return h
      
