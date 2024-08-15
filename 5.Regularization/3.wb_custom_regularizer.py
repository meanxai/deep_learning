# [MXDL-5-01] 3.wb_custom_regularizer.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/WhN0ppyFLq0
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Read a saved dataset
with open('data/blobs.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

# Create an ANN model.
n_input = x_train.shape[1]  # number of input neurons
n_output = 1                # number of output neurons
n_hidden = 32               # number of hidden neurons
R = 0.01                    # regularization constant
adam = optimizers.Adam(learning_rate=0.005)

# Custom regularizer for L3 regularization
# L3 regularization is rarely used, but if you want to use 
# it for some reason, you can implement it using a custom 
# regularizer like this.
class reg_L3(regularizers.Regularizer):
    def __init__(self, reg_lambda):
        self.R = reg_lambda
    
    def __call__(self, x):
        # The w or b of a layer is passed to x.
        return self.R * tf.reduce_sum(tf.math.pow(tf.math.abs(x), 3))
    
# The data is simple, but we intentionally added many hidden 
# layers to the model to demonstrate the effect of regularization. 
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation = 'relu', 
          kernel_regularizer=reg_L3(R),
          bias_regularizer=reg_L3(R))(x_input)

# 4 more hidden layers
for i in range(4):
    h = Dense(n_hidden, activation = 'relu', 
              kernel_regularizer=reg_L3(R),
              bias_regularizer=reg_L3(R))(h)

y_output = Dense(n_output, activation='sigmoid',
                 kernel_regularizer=reg_L3(R),
                 bias_regularizer=reg_L3(R))(h)

model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', 
              optimizer = adam)

h = model.fit(x_train, y_train, epochs=100, batch_size=50,
              validation_data=[x_test, y_test])

# Visually see the loss history
plt.plot(h.history['loss'], c='blue', label='train loss')
plt.plot(h.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:4f}".format(acc))

# https://github.com/keras-team/keras/blob/master/keras/regularizers.py
# class Regularizer:
#   """Regularizer base class"""
#   def __call__(self, x):
#       """Compute a regularization penalty from an input tensor."""
#       return 0.0
# Since there is nothing in Regularizer's __call__(), we define __call__().
# class reg_L3(regularizers.Regularizer):
#     def __init__(self, reg_lambda):
#         self.R = reg_lambda
    
#     def __call__(self, x):
#         # The w or b of a layer is passed to x.
#         return self.R * tf.reduce_sum(tf.math.pow(tf.math.abs(x), 3))