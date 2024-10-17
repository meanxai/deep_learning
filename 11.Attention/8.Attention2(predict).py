# [MXDL-11-04] 7.Attention2(predict).py (Luong's version)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/eyXHpL4dlYU
#
from tensorflow.keras.layers import Input, Dense, TimeDistributed
from Attention import Encoder, Decoder
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	data, _, _, _ = pickle.load(f)      

n_hidden = 100
n_emb = 30
n_feed = 30
n_step = 50
n_feat = data.shape[1]

# Time series embedding layer.
EmbedInput = Dense(n_emb, activation='tanh')

# Trained Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
e_enc = EmbedInput(i_enc)
o_enc, h_enc = Encoder(n_hidden)(e_enc)

# Trained Decoder
i_dec = Input(batch_shape=(None, n_step, n_feat))
e_dec = EmbedInput(i_dec)
o_dec = Decoder(n_hidden, n_feed)(e_dec, o_enc, h_enc) 
y_dec = TimeDistributed(Dense(n_feat))(o_dec)

model = Model([i_enc, i_dec], y_dec)
model.load_weights("models/attention2.h5")

# prediction
n_future = 50
e_data = data[-50:].reshape(-1, 50, 2)
d_data = np.zeros(shape=(1, 50, 2))
d_data[0, 0, :] = data[-1]

for i in range(n_future):
    y_hat = model.predict([e_data, d_data], verbose=0)
    y_hat = y_hat[0, :, :]  # remove the first dimension
    
    if i < n_future - 1:
        d_data[0, i+1, :] = y_hat[i, :]
    
    print(i+1, ':', y_hat[i, :])

# Plot the past time series and the predicted future time series.
y_past = data[-100:]
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(len(y_past), len(y_past) + len(y_hat))
plt.plot(ax1, y_past[:, 0], '-o', c='blue', markersize=3, 
         label='Original time series 1', linewidth=1)
plt.plot(ax1, y_past[:, 1], '-o', c='red', markersize=3, 
         label='Original time series 2', linewidth=1)
plt.plot(ax2, y_hat[:, 0], '-o', c='green', markersize=3,
         label='Predicted time series 1')
plt.plot(ax2, y_hat[:, 1], '-o', c='orange', markersize=3, 
         label='Predicted time series 2')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()



