# [MXDL-13-02] 6.denoise_timeseries.py
# Denoising time series data using LSTM autoencoder
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Q8AyijWiJyk
#
import numpy as np
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

# Read noisy time series and split them into training and test data
with open('data/noisy_sine.pkl', 'rb') as f:
        x, y = pickle.load(f)

x_train, x_test, y_train, y_test = train_test_split(x, y)
n_timesteps = x.shape[1]
n_features = x.shape[2]

# Build a denoising LSTM autoencoder
x_input = Input(batch_shape=(None, *x_train.shape[1:]))
h_enc = LSTM(100)(x_input)
h_enc = RepeatVector(n_timesteps)(h_enc)
h_dec = LSTM(100, return_sequences=True)(h_enc)
y_output = TimeDistributed(Dense(n_features))(h_dec)

model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(lr=0.005))
model.summary()

# Training
h = model.fit(x_train, y_train, epochs = 100, batch_size=300,
              validation_data=[x_test, y_test])

# Loss History
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.show()

# Denoising x_test
x_pred = model.predict(x_test, verbose=0)

# Visualize
n = 2
idx = np.random.choice(x_test.shape[0], n)[0]
fig, ax = plt.subplots(n, 3, figsize=(12, 5))

for i in range(n):
    ax[i, 0].plot(x_test[i])
    ax[i, 1].plot(x_pred[i])
    ax[i, 2].plot(y_test[i])

    if i == 0:
        ax[i, 0].set_title("Noisy sine curves (x_test)")
        ax[i, 1].set_title("Denoised sine curves (x_pred)")
        ax[i, 2].set_title("Original sine curves (y_test)")
plt.show()