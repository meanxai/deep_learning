# [MXDL-11-02] 2.seq2seq(train).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/KdQmVoEsBoE
#
from tensorflow.keras.layers import Input, GRU, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	_, xi_enc, xi_dec, xp_dec = pickle.load(f)

n_hidden = 100
n_step = xi_enc.shape[1]  # 50
n_feat = xi_enc.shape[2]  # 2
     
# Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
h_enc = GRU(n_hidden)(i_enc)

# Decoder
i_dec = Input(batch_shape=(None, n_step, n_feat))
o_dec = GRU(n_hidden, 
            return_sequences=True)(i_dec, initial_state = h_enc)
y_dec = TimeDistributed(Dense(n_feat))(o_dec)

model = Model([i_enc, i_dec], y_dec)
model.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=0.001))
model.summary()

# Training: teacher forcing
hist = model.fit([xi_enc, xi_dec], xp_dec, 
                 batch_size=200, epochs=100)
# Save the trained model
model.save_weights("models/seq2seq.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
