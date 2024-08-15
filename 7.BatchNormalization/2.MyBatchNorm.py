# [MXDL-7-02] 2.MyBatchNormalization.py
# Custom BatchNormalization layer
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/_qeKY0I32nk
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Generate a dataset
# x, y = make_blobs(n_samples=1000, n_features=2, 
#                   centers=[[0., 0.], [0.5, 0.1]], 
#                   cluster_std=0.25, center_box=(-1., 1.))
# y = y.reshape(-1, 1).astype('float32')
# x_train, x_test, y_train, y_test = train_test_split(x, y)
# with open('data/blobs.pkl', 'wb') as f:
#     pickle.dump([x_train, x_test, y_train, y_test], f)

with open('data/blobs.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
    
# Visually see the data
plt.figure(figsize=(5,4))
color = [['red', 'blue'][a] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=20, c=color, alpha=0.3)
plt.show()

# Custom Batch Normalization layer
class MyBatchNorm(Layer):
    def __init__(self, units=32, rho=0.99):
        super(MyBatchNorm, self).__init__()
        self.units = units        
        self.rho = rho
    
    def build(self, input_shape):
        dim = (input_shape[-1],)
        self.gamma = self.add_weight(
            shape=dim, name='gamma',
            initializer='ones',
            trainable=True)
        
        self.beta = self.add_weight(
            shape=dim, name='beta',
            initializer='zeros',
            trainable=True)
        
        self.mean_ema = self.add_weight(
            shape=dim, name='mean_ema',
            initializer='zeros',
            trainable=False)
        
        self.var_ema = self.add_weight(
            shape=dim, name='var_ema',
            initializer='zeros',
            trainable=False)
        
    def call(self, inputs, training=True):
        if training:
            mean = tf.reduce_mean(inputs, axis=0)
            var = tf.math.reduce_variance(inputs, axis=0)
        else:
            mean = self.mean_ema
            var = self.var_ema
            print("\n===========> training=False")
        h_hat = (inputs - mean) / tf.math.sqrt(var + EPSILON)
        
        self.mean_ema.assign(self.rho * self.mean_ema + (1 - self.rho) * mean)
        self.var_ema.assign(self.rho * self.var_ema + (1 - self.rho) * var)

        return self.gamma * h_hat + self.beta
    
# Create an ANN model with Batch Normalization
n_input = x_train.shape[1] # number of input neurons
n_output = 1               # number of output neurons
n_hidden = 128             # number of hidden neurons
EPSILON = 1e-5
BATCH_SIZE = 300
RHO = 0.99
adam = optimizers.Adam(learning_rate=0.01)

BatchNorm = MyBatchNorm(n_hidden, RHO)
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, use_bias=False)(x_input)
h = BatchNorm(h)
h = Activation('relu')(h)

y_output = Dense(n_output, activation='sigmoid')(h)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=adam)

# training        
h = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              epochs=50, batch_size=300)

# Visually see the loss history
plt.plot(h.history['loss'], c='blue', label='train loss')
plt.plot(h.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of test data
y_hat = model(x_test, training=False).numpy()
y_pred = (y_hat > 0.5) * 1
acc = (y_pred == y_test).mean()
print("The accuracy of test data = {:4f}".format(acc))

BatchNorm.gamma
BatchNorm.beta
BatchNorm.mean_ema
BatchNorm.var_ema
