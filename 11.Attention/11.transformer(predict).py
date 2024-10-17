# [MXDL-11-06] 11.transformer(predict).py
# Original transformer code: https://github.com/suyash/transformer
# The original transformer code above is for natural language 
# processing. I modified it slightly for use in time series 
# forecasting.
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/KkfAvhmpznM
#
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from transformer import Encoder, Decoder
from transformer import PaddingMask, PaddingAndLookaheadMask
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	data, _, _, _ = pickle.load(f)  
     
n_tstep = 50
n_feat = data.shape[1]   # 2
d_model = 100

# Trained Encoder
EmbDense = Dense(d_model, use_bias=False)
i_enc = Input(batch_shape=(None, n_tstep, n_feat))
h_enc = EmbDense(i_enc)
padding_mask = PaddingMask()(h_enc)
encoder = Encoder(num_layers = 1, 
                  d_model = d_model, 
                  num_heads = 5, 
                  d_ff = 64, 
                  dropout_rate=0.5)
o_enc, _ = encoder(h_enc, padding_mask)


# Trained Decoder
i_dec = Input(batch_shape=(None, None, n_feat))
h_dec = EmbDense(i_dec)
lookahead_mask = PaddingAndLookaheadMask()(h_dec)
decoder = Decoder(num_layers = 1, 
                  d_model = d_model, 
                  num_heads = 5, 
                  d_ff = 64,
                  dropout_rate=0.5)
o_dec, _, _ = decoder(h_dec, o_enc, lookahead_mask, padding_mask)
y_dec = Dense(n_feat)(o_dec)

model = Model(inputs=[i_enc, i_dec], outputs=y_dec)
model.load_weights("models/transformer.h5")

m = Model(i_dec, h_dec)

# prediction
n_future = 50
e_data = data[-n_tstep:].reshape(-1, n_tstep, n_feat)
d_data = np.zeros(shape=(1, n_future, n_feat))
d_data[0, 0, :] = data[-1]

for i in range(n_future):
    y_hat = model.predict([e_data, d_data], verbose=0)
    
    if i < n_future - 1:
        d_data[0, i+1, :] = y_hat[0, i, :]
    
    print(i+1, ':', y_hat[0, i, :])

# Plot the past time series and the predicted future time series.
y_past = data[-100:]
y_hat = np.vstack([y_past[-1], y_hat[0,:,:]])

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

