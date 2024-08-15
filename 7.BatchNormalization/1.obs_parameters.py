# [MXDL-7-02] 1.obs_parameters.py
# Observing the parameters inside Batch Normalization layer
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/jwyQxTFpHzk
#
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model

x = np.random.normal(size=(100, 3))
y = np.random.choice([0,1], 100).reshape(-1,1)
e = 0.001; rho = 0.99

x_input = Input(batch_shape=(None, 3))
h = Dense(4, use_bias=False, name='HN')(x_input)
r = BatchNormalization(momentum=rho, epsilon=e, name = 'BN')(h)
h_act = Activation('relu')(r)
y_output = Dense(1, activation='sigmoid')(h_act)
model = Model(x_input, y_output)
model.compile(loss='mean_squared_error', optimizer = 'adam')
model_h = Model(x_input, h)
model_r = Model(x_input, r)
model.summary()

# Initial values of the parameters in Batch Normalization layer
gamma, beta, mu, var = model.get_layer('BN').get_weights()
print('\nInitial values:')
print('   γ =', gamma.round(3))
print('   β =', beta.round(3))
print('E[h] =', mu.round(3))
print('V[h] =', var.round(3))

# Training. Gamma and beta are also learned, and moving mu and
# var are calculated and stored.
model.fit(x, y, epochs=10, batch_size=10, verbose=0)

# outputs of the hidden layer
print('After training: prediction stage')
ho = model_h.predict(x, verbose=0)[:3]
print('h = '); print(ho.round(3))

# outputs of the Batch Normalization layer
ro = model_r.predict(x, verbose=0)[:3]
print('\nr = '); print(ro.round(3))

# Parameters stored in Batch Normalization
# layer
gamma, beta, mu, var = \
    model.get_layer('BN').get_weights()
print('\n   γ =', gamma.round(3))
print('   β =', beta.round(3))
print('E[h] =', mu.round(3))
print('V[h] =', var.round(3))

# Let's manually calculate the outputs of 
# the BatchNormalization layer.
rm = gamma * (ho - mu) / np.sqrt(var + e) + beta
print('\nManual calculation (r)')
print(rm.round(3)) # This matches r above.

