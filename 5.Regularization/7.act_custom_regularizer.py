# [MXDL-5-02] 7.act_custom_regularizer.py
# Imposing a sparsity constraint on the hidden neurons 
# by activity regularization
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/nhLFkUbpFfA
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
rho = 0.5                   # desired output of hidden layer
adam = optimizers.Adam(learning_rate=0.001)
L2 = regularizers.L2(0.01)

# Custom regularizer for KL divergence regularization
class h_constrined(regularizers.Regularizer):
    def __init__(self, reg_lambda, rho):
        self.R = reg_lambda
        self.rho = rho
    
    def __call__(self, x):
        rho_hat = tf.reduce_mean(x, axis=0)
        rho_hat = tf.clip_by_value(rho_hat, 1e-6, 0.99999)
        kl = self.rho * tf.math.log(self.rho / rho_hat) + \
             (1. - self.rho) * tf.math.log((1. - self.rho) / (1. - rho_hat))
        return self.R * tf.reduce_sum(kl)
    
# The data is simple, but we intentionally added many hidden 
# layers to the model to demonstrate the effect of regularization. 
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation='relu', 
          activity_regularizer=L2)(x_input)

# 3 more hidden layers
for i in range(3):
    h = Dense(n_hidden, activation='relu', 
              activity_regularizer=L2)(h)

# last hidden layer
h_last = Dense(n_hidden, activation='relu', 
                activity_regularizer=h_constrined(0.1, rho))(h)

y_output = Dense(n_output, activation='sigmoid')(h_last)

model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', 
              optimizer = adam)

h_model = Model(x_input, h_last)

f = model.fit(x_train, y_train, epochs=100, batch_size=50,
              validation_data=[x_test, y_test])

# Visually see the loss history
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:4f}".format(acc))

# Check the output distribution of the last hidden neuron.
h_act = h_model.predict(x_test).reshape(-1,)
plt.hist(h_act, bins=50)
plt.show()
print("Mean of the activations =", h_act.mean())
