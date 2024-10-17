# [MXDL-11-06] 10.transformer(train).py
# Transformer code: https://github.com/suyash/transformer
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
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pickle
from transformer import Encoder, Decoder
from transformer import PaddingMask, PaddingAndLookaheadMask

# Read dataset
with open('dataset.pkl', 'rb') as f:
 	_, xi_enc, xi_dec, xp_dec = pickle.load(f)

n_tstep = xi_enc.shape[1]  # 50
n_feat = xi_enc.shape[2]   # 2
d_model = 100

# Encoder
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

# Decoder
i_dec = Input(batch_shape=(None, n_tstep, n_feat))
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
model.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=0.001))

# Training: teacher forcing
hist = model.fit([xi_enc, xi_dec], xp_dec, 
                 epochs=100,
                 batch_size = 200)

# Save the trained model
# model.save("models/transformer.h5")

# Visually see the loss history
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

