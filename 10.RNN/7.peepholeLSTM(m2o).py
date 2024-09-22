# [MXDL-10-06] 7.peepholeLSTM(m2o).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/wCspLrRv_Ts
#
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
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

n_hidden = 50   # the number of
                # hidden units

# peephole LSTM custom layer
class PeepholeLSTM(Layer):
    def __init__(self, n_hidden):
        super().__init__()
        self.nh = n_hidden
      
        # weights and bias for x
        self.Wf = Dense(n_hidden)
        self.Wi = Dense(n_hidden)
        self.Wc = Dense(n_hidden)
        self.Wo = Dense(n_hidden)
		
        # weights for h. The biases are included in w above.
        self.Rf = Dense(n_hidden, use_bias=False)  # forget
        self.Ri = Dense(n_hidden, use_bias=False)  # input
        self.Rc = Dense(n_hidden, use_bias=False)  # candidate state
        self.Ro = Dense(n_hidden, use_bias=False)  # output

        # peephole connections
        self.Pf = Dense(n_hidden, use_bias=False)
        self.Pi = Dense(n_hidden, use_bias=False)
        self.Po = Dense(n_hidden, use_bias=False)

    def lstm_cell(self, x, h, c):
        f_gate=tf.math.sigmoid(self.Wf(x) + self.Rf(h) + self.Pf(c))
        i_gate=tf.math.sigmoid(self.Wi(x) + self.Ri(h) + self.Pi(c))
        c_tild= tf.math.tanh(self.Wc(x) + self.Rc(h))
        c_stat = c * f_gate + c_tild * i_gate
        o_gate=tf.math.sigmoid(self.Wo(x)+self.Ro(h)+self.Po(c_stat))
        h_stat = tf.math.tanh(c_stat) * o_gate
        return h_stat, c_stat

    def call(self, x):
        # initialize h, c state
        h = tf.zeros(shape=(tf.shape(x)[0], self.nh))
        c = tf.zeros(shape=(tf.shape(x)[0], self.nh))

        # Repeat lstm_cell for the number of time steps
        for t in range(x.shape[1]):
            h, c = self.lstm_cell(x[:, t, :], h, c)
        return h

# Build a peephole LSTM model
n_feat = x_train.shape[-1]
n_output = y_train.shape[-1]
x_input = Input(batch_shape=(None, n_step, n_feat))
h = PeepholeLSTM(n_hidden)(x_input)
y_output = Dense(n_output)(h)
model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary()   # Trainable params: 18,202

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
         label='Estimated time series 1')
plt.plot(ax2, f[:, 1], '-o', c='orange', markersize=3, 
         label='Estimated time series 2')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
      