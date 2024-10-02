# [MXDL-11-03] 6.Attention1(predict).py (Simple version)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/k2iDqmp4Oeo
#
from tensorflow.keras.layers import Input, GRU, Dense, Dot
from tensorflow.keras.layers import Activation, Concatenate
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
n_emb = 30

def AttentionLayer(d, e):
    dot_product = Dot(axes=(2, 2))([d, e])
    score = Activation('softmax')(dot_product)
    value = Dot(axes=(2, 1))([score, e])
    return Concatenate()([value, d])

# Time series embedding layer.
EmbedInput = Dense(n_emb, activation='tanh')

# Trained Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
e_enc = EmbedInput(i_enc)
o_enc, h_enc = GRU(n_hidden, 
                   return_sequences=True,
                   return_state = True)(i_enc)

# Trained Decoder
# Layers to be shared in the prediction model
TrainedGRU = GRU(n_hidden,
                    return_sequences=True,
                    return_state=True)
TrainedFFN = TimeDistributed(Dense(n_feat))

i_dec = Input(batch_shape=(None, 1, n_feat))
e_dec = EmbedInput(i_dec)
o_dec, _ = TrainedGRU(i_dec, initial_state = h_enc)
a_dec = AttentionLayer(o_dec, o_enc)
y_dec = TrainedFFN(a_dec)

model = Model([i_enc, i_dec], y_dec)
model.load_weights("models/attention1.h5")

# Encoder model for prediction
Encoder = Model(i_enc, [o_enc, h_enc])

# Decoder model for prediction
i_stat = Input(batch_shape = (None, n_hidden))
i_henc = Input(batch_shape = (None, n_step, n_hidden))
o_pre, h_pre = TrainedGRU(i_dec, initial_state = i_stat)
a_pre = AttentionLayer(o_pre, i_henc)
y_pre = TrainedFFN(a_pre)
Decoder = Model([i_dec, i_stat, i_henc], [y_pre, h_pre])

# prediction
e_seed = data[-50:].reshape(-1, 50, 2)
d_seed = data[-1].reshape(-1, 1, 2)
oe, he = Encoder.predict(e_seed, verbose=0)

n_future = 50
y_pred = []
for i in range(n_future):
    yd, hd = Decoder.predict([d_seed, he, oe], verbose=0)
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


