# [MXDL-13-03] 7.sparse_autoencoder.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/_61lTOVB6xA
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer, L1
import matplotlib.pyplot as plt
import pickle

# Read a MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)

x = x.reshape(-1, 28, 28, 1)
n_train = int(0.9 * x.shape[0])
x_train = x[:n_train]
x_test = x[n_train:]

# Custom regularizer for KL divergence regularization
class sparsity_constrint(Regularizer):
    def __init__(self, reg_lambda, rho):
        self.R = reg_lambda
        self.rho = rho
    
    def __call__(self, q):
        rh = tf.reduce_mean(q, axis=0)
        rh = tf.clip_by_value(rh, 1e-6, 0.99999)
        r = self.rho
        KL = r * tf.math.log(r / rh) + (1-r)*tf.math.log((1-r)/(1-rh))
        return self.R * tf.reduce_sum(KL)

rho = 0.1          # sparsity parameter
n_filters = 10

# Build a sparse autoencoder model
x_input = Input(batch_shape=(None, 28, 28, 1))

# Encoder
e_conv = Conv2D(n_filters, (3,3), strides=1, 
                padding = 'same', activation='relu',
                activity_regularizer=sparsity_constrint(0.01, rho))(x_input)

# e_conv = Conv2D(n_filters, (3,3), strides=1, 
#                 padding = 'same', activation='relu',
#                 activity_regularizer=L1(0.00001))(x_input)

# Decoder
d_conv = Conv2DTranspose(n_filters, (3,3), strides=1, 
                padding = 'same', activation='relu')(e_conv)
x_output = Conv2D(filters=1, kernel_size=(3,3), strides=1, 
                padding = 'same', activation='sigmoid')(d_conv)

model = Model(x_input, x_output)
encoder = Model(x_input, e_conv)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005))
model.summary()

# Training
hist = model.fit(x_train, x_train, epochs=100, batch_size=100,
                validation_data=[x_test, x_test])

# Loss history
plt.plot(hist.history['loss'], c='blue', label='train loss')
plt.plot(hist.history['val_loss'], c='red', label='test loss')
plt.legend()
plt.show()

# Visualize the encoder's outputs for some test data points.
n_sample = 10
s_test = np.take(x_test, np.arange(0,10), axis=0)
s_pred = encoder.predict(s_test)

fig, ax = plt.subplots(1, n_sample, figsize=(12,4))
for i in range(n_sample):
    ax[i].imshow(s_test[i], cmap='gray')
    ax[i].axis('off')
plt.show()

fig, ax = plt.subplots(n_filters, n_sample, figsize=(12,12))
for i in range(n_filters):
    for k in range(n_sample):
        ax[i, k].imshow(s_pred[k][:, :, i], cmap='gray')
        ax[i, k].axis('off')
plt.show()





