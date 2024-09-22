# [MXDL-10-07] 9.GRU(m2m).py (Custom GRU layer)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/aXnmiadWfLg
#
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Generate training data: 2 noisy sine curves
n = 1000        # the number of data points
n_step = 20     # the number of time steps
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T
m = np.arange(0, n - n_step)
x_train = np.array([data[i:(i+n_step), :] for i in m])
y_train = np.array([data[(i+1):(i+1+n_step), :] for i in m])

n_hidden = 50   # the number of
                # hidden units
                
# GRU custom layer
class MyGRU(Layer):
    def __init__(self, n_hidden, return_sequences=False):
        super().__init__()
        self.nh = n_hidden
        self.return_sequences = return_sequences

        # weights and bias for data x
        self.Wr = Dense(n_hidden)  # reset gate
        self.Wz = Dense(n_hidden)  # update gate
        self.Wg = Dense(n_hidden)  # candidate
		
        # weights for hidden state h. The biases are included in w above.
        self.Hr = Dense(n_hidden, use_bias=False)  # reset
        self.Hz = Dense(n_hidden, use_bias=False)  # update
        self.Hg = Dense(n_hidden, use_bias=False)  # candidate

    def gru_cell(self, x, h):
        rt = tf.math.sigmoid(self.Wr(x) + self.Hr(h))
        zt = tf.math.sigmoid(self.Wz(x) + self.Hz(h))
        gt = tf.math.tanh(self.Wg(x) + self.Hg(rt * h))
        h_stat = (1. - zt) * h + zt * gt
        return h_stat

    def call(self, x):
        # initialize h
        h = [tf.zeros(shape=(tf.shape(x)[0], self.nh))]
        
        # Repeat gru_cell for the number of time steps
        for t in range(x.shape[1] - 1):
            ht = self.gru_cell(x[:, t, :], h[-1])
            h.append(ht)
            
        if self.return_sequences:
            h = tf.convert_to_tensor(h)         # (20, None, 50)
            h = tf.transpose(h, perm=[1, 0, 2]) # (None, 20, 50)
            return h        # return whole h
        else:
            return h[-1]    # return final h    

# Build a GRU model
n_feat = x_train.shape[-1]
n_output = y_train.shape[-1]

x_input = Input(batch_shape=(None, n_step, n_feat))
h = MyGRU(n_hidden, return_sequences=True)(x_input)
y_output = TimeDistributed(Dense(n_output))(h)
model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary()

# Training
hist = model.fit(x_train, y_train, epochs=50, batch_size=50)

# Visually see the loss history
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
last_data = data[-n_last:]  # The last n_last data points
for i in range(n_future):
    # Predict the next value with the last n_step data points.
    px = last_data[-n_step:, :].reshape(1, n_step, 2)

    # Predict the next value
    y_hat = model.predict(px, verbose=0)[:, -1, :]
    
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
         label='Estimated time series 1', linewidth=1)
plt.plot(ax2, f[:, 1], '-o', c='orange', markersize=3, 
         label='Estimated time series 2', linewidth=1)
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()


