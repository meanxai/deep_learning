# [MXDL-4-02] 7.custom_loss.py
# Applying L2 regularization in loss function
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/oemrJonU-tE
#
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Generate a data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + \
    np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, test data set
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_pred = np.linspace(0, 1, 200).reshape(-1, 1)

# Custom loss: Applying L2 regularization
class regularized_loss(tf.keras.losses.Loss):
    def __init__(self, C, h_layer, o_layer):
        super(regularized_loss, self).__init__()
        self.C = C
        self.h_layer = h_layer
        self.o_layer = o_layer
        
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_true - y_pred))
        
        wh = self.h_layer.weights[0] # weights in hidden layer
        wo = self.o_layer.weights[0] # weights in output layer
        mse += self.C * tf.reduce_sum(tf.math.square(wh))
        mse += self.C * tf.reduce_sum(tf.math.square(wo))
        return mse

# Create an ANN model
n_input = x.shape[1]  # number of input neurons
n_output = 1          # number of output neurons
n_hidden = 8          # number of hidden neurons
adam = optimizers.Adam(learning_rate=0.01)

h_layer = Dense(n_hidden, activation='tanh')   # hidden layer
o_layer = Dense(n_output, activation='linear') # output layer

x_input = Input(batch_shape=(None, n_input))
h = h_layer(x_input)
y_output = o_layer(h)
model = Model(x_input, y_output)

myloss = regularized_loss(0.00, h_layer, o_layer)
model.compile(loss=myloss, optimizer=adam)

# Training       
f = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              epochs=200, batch_size=50)

# Visually see the data.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=50, c='blue', alpha=0.5, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.5, label='valid')
plt.legend()
plt.show()

# Visually see the loss history.
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Visually see the prediction result.
y_pred = model.predict(x_pred)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.3, label='validation')
plt.scatter(x_pred, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()


