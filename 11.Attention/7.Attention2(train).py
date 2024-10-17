# [MXDL-11-04] 6.Attention(train).py (Luong's version)
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
 	_, xi_enc, xi_dec, xp_dec = pickle.load(f)     

n_hidden = 100
n_emb = 30
n_feed = 30
n_step = xi_enc.shape[1]
n_feat = xi_enc.shape[2]

# Time series embedding layer.
EmbedInput = Dense(n_emb, activation='tanh')

# Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
e_enc = EmbedInput(i_enc)
o_enc, h_enc = Encoder(n_hidden)(e_enc)

# Decoder
i_dec = Input(batch_shape=(None, n_step, n_feat))
e_dec = EmbedInput(i_dec)
o_dec = Decoder(n_hidden, n_feed)(e_dec, o_enc, h_enc) 
y_dec = TimeDistributed(Dense(n_feat))(o_dec)

model = Model([i_enc, i_dec], y_dec)
model.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=0.001))
   
# Training: teacher forcing
hist = model.fit([xi_enc, xi_dec], xp_dec, 
                 batch_size=500, epochs=200)

# Save the trained model
model.save_weights("models/attention2.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

