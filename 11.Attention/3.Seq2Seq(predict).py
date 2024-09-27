# [MXDL-11-02] 3.seq2seq(predict).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/KdQmVoEsBoE
#
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	data, _, _, _ = pickle.load(f)

n_hidden = 100
n_step = 50
n_feat = 2

# Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
h_enc = GRU(n_hidden)(i_enc)

# Decoder
# Instantiate a single-step GRU and a many-to-many output layer classes 
# so that they can be shared later in the prediction decoder model.
SingleStepGRU = GRU(n_hidden, return_sequences=True, return_state=True)
ManyOUT = TimeDistributed(Dense(n_feat))

i_dec = Input(batch_shape=(None, 1, n_feat))
o_dec, _ = SingleStepGRU(i_dec, initial_state = h_enc)
y_dec = ManyOUT(o_dec)
model = Model([i_enc, i_dec], y_dec)
model.load_weights("models/seq2seq.h5")

# Encoder model for time series forecasting.
Encoder = Model(i_enc, h_enc)

# Decoder for time series forecasting: single-step model
i_pre = Input(batch_shape = (None, n_hidden))
o_pre, h_pre = SingleStepGRU(i_dec, initial_state = i_pre)
y_pre = ManyOUT(o_pre)
Decoder = Model([i_dec, i_pre], [y_pre, h_pre])

# prediction
e_seed = data[-50:].reshape(-1, 50, 2)
d_seed = data[-1].reshape(-1, 1, 2)
he = Encoder.predict(e_seed, verbose=0)

n_future = 50
y_pred = []
for i in range(n_future):
    yd, hd = Decoder.predict([d_seed, he], verbose=0)
    y_pred.append(yd.reshape(2,))
    
    he = hd
    d_seed = yd
y_pred = np.array(y_pred)

# Plot the past time series and the predicted future time series.
y_past = data[-100:]
plt.figure(figsize=(12, 6))
ax1 = np.arange(1, len(y_past) + 1)
ax2 = np.arange(len(y_past), len(y_past) + len(y_pred))
plt.plot(ax1, y_past[:, 0], '-o', c='blue', markersize=3, 
         label='Original time series 1', linewidth=1)
plt.plot(ax1, y_past[:, 1], '-o', c='red', markersize=3, 
         label='Original time series 2', linewidth=1)
plt.plot(ax2, y_pred[:, 0], '-o', c='green', markersize=3,
         label='Predicted time series 1')
plt.plot(ax2, y_pred[:, 1], '-o', c='orange', markersize=3, 
         label='Predicted time series 2')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
