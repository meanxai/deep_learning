# [MXDL-10-03] 1.simplernn(m2o).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/9Zo2yIDUHDM
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Generate training data: 2 noisy sine curves
n = 1000        # the number of data points
n_step = 20     # the number of time steps
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T  # shape = (1000, 2)

m = np.arange(0, n - n_step)
x_train = np.array([data[i:(i+n_step), :] for i in m])
y_train = np.array([data[i, :] for i in (m + n_step)])

# Create a SimpleRNN model
n_feat = x_train.shape[-1]    # 2
n_output = y_train.shape[-1]  # 2
n_hidden = 50

# SimpleRNN class
class MySimpleRNN(Layer):
   # nf: the number of features, nh: the number of hidden units
   def __init__(self, nf, nh):  
      super().__init__()
      self.nh = nh
      w_init = initializers.GlorotUniform()
      b_init = tf.zeros_initializer()
      self.wx = tf.Variable(w_init([nf, nh]), trainable = True)
      self.wh = tf.Variable(w_init([nh, nh]), trainable = True)
      self.b = tf.Variable(b_init([1, nh]), trainable = True)
		
   def call(self, x):
      h = tf.zeros(shape=(tf.shape(x)[0], self.nh)) # initial values
      for t in range(tf.shape(x)[1]): # Recurrence
         # shape: [None, nf]*[nf, nh] + [None, nh]*[nh, nh] + [1, nh]
         z = tf.matmul(x[:, t, :], self.wx) + \
             tf.matmul(h, self.wh) + self.b
         h = tf.math.tanh(z)
      return h

# Create a SimpleRNN model
x_input = Input(batch_shape=(None, n_step, n_feat))
h = MySimpleRNN(n_feat, n_hidden)(x_input)
y_output = Dense(n_output)(h)
model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary()  # trainable parameters = 2,752

# Training
hist = model.fit(x_train, y_train, epochs=50, batch_size=50)

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Predict future values for the next 50 periods.
# After predicting the next value, re-enter the predicted value 
# to predict the next value. Repeat this process 50 times.
n_future = 50
n_last = 100
last_data = data[-n_last:]  # The last n_last data points of the original data.
for i in range(n_future):
    # Predict the next value with the last n_step data points.
    px = last_data[-n_step:, :].reshape(1, n_step, 2)

    # Predict the next value
    y_hat = model.predict(px, verbose=0)
    
    # Append the predicted value ​​to the last_data array.
    # In the next iteration, the predicted value is input 
    # along with the existing data points.
    last_data = np.vstack([last_data, y_hat])

p = last_data[:-n_future, :]        # past time series
f = last_data[-(n_future + 1):, :]  # future time series

# Plot past and future time series.
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(p) + 1)
ax2 = np.arange(len(p), len(p) + len(f))
plt.plot(ax1, p[:, 0], '-o', c='blue', markersize=3, 
         label='Actual time series 1', linewidth=1)
plt.plot(ax1, p[:, 1], '-o', c='red', markersize=3, 
         label='Actual time series 2', linewidth=1)
plt.plot(ax2, f[:, 0], '-o', c='green', markersize=3,
         label='Predicted time series 1')
plt.plot(ax2, f[:, 1], '-o', c='orange', markersize=3,
         label='Predicted time series 2')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
