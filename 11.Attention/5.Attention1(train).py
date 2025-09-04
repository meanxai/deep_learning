import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [MXDL-11-03] 5.Attention1(train).py (Simple version)
from tensorflow.keras.layers import Input, GRU, Dense, Dot
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	_, xi_enc, xi_dec, xp_dec = pickle.load(f)     

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

# Encoder
i_enc = Input(batch_shape=(None, n_step, n_feat))
e_enc = EmbedInput(i_enc)
o_enc, h_enc = GRU(n_hidden, 
                   return_sequences=True,
                   return_state = True)(e_enc)

# Decoder
i_dec = Input(batch_shape=(None, n_step, n_feat))
e_dec = EmbedInput(i_dec)
o_dec = GRU(n_hidden, 
            return_sequences=True)(e_dec, initial_state = h_enc)
a_dec = AttentionLayer(o_dec, o_enc)
y_dec = TimeDistributed(Dense(n_feat))(a_dec)

model = Model([i_enc, i_dec], y_dec)
model.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=0.001))
   
# Training: teacher forcing
hist = model.fit([xi_enc, xi_dec], xp_dec, 
                 batch_size=500, epochs=200)

# Save the trained model
model.save_weights("models/attention1.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

