# [MXDL-8-02] 2.xavier_glorot.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/M8jZWt0bI14
#
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

method = 'Normal'
method = 'a'
if method == 'Normal':
    init_w1 = initializers.GlorotNormal()
    init_w2 = initializers.GlorotNormal()
else:
    init_w1 = initializers.GlorotUniform()
    init_w2 = initializers.GlorotUniform()

n_in = 50
n_s1 = 80
n_s2 = 100
x = np.random.normal(size=(1000, n_in))

# Create an ANN model
x_input = Input(batch_shape=(None, n_in))
s1 = Dense(n_s1, kernel_initializer=init_w1, activation='tanh',
           name='W1')(x_input)
s2 = Dense(n_s2, kernel_initializer=init_w2, activation='tanh',
           name='W2')(s1)
y_output = Dense(1, activation='sigmoid')(s2)
s1_model = Model(x_input, s1)
s2_model = Model(x_input, s2)
model = Model(x_input, y_output)

w1 = model.get_layer('W1').get_weights()[0].reshape(-1,)
w2 = model.get_layer('W2').get_weights()[0].reshape(-1,)

s1_out = s1_model.predict(x, verbose=0).reshape(-1,)
s2_out = s2_model.predict(x, verbose=0).reshape(-1,)

plt.hist(x.reshape(-1,), bins=50); plt.show()
plt.hist(s1_out, bins=50); plt.show()
plt.hist(s2_out, bins=50); plt.show()

if method == 'Normal':
    print('[Keras  ] σ of w1 = {:.3f}'.format(w1.std()))
    print('[Formula] σ of w1 = {:.3f}'.\
          format(np.sqrt(2 / (n_in + n_s1))))
else:
    print('[Keras  ] a of w1 = {:.3f} ~ {:.3f}'.\
          format(w1.min(), w1.max()))
    print('[Formula] a of w1 = ±{:.3f}'.\
          format(np.sqrt(6 / (n_in + n_s1))))

if method == 'Normal':
    print('\n[Keras  ] σ of w2 = {:.3f}'.format(w2.std()))
    print('[Formula] σ of w2 = {:.3f}'.
          format(np.sqrt(2 / (n_s1 + n_s2))))
else:
    print('\n[Keras] a of w2 = {:.3f} ~ {:.3f}'.
          format(w2.min(), w2.max()))
    print('[Formula] a of w2 = ±{:.3f}'.
          format(np.sqrt(6 / (n_s1 + n_s2))))
