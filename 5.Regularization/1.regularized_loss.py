# [MXDL-5-01] 1.regularized_loss.py
# Regularization can be easily implemented using Keras Dense's 
# kernel_regularizer and bias_regularizer, but to better 
# understand how regularization works, we implement it by 
# creating a regularized loss function.
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/WhN0ppyFLq0
#
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Generate a dataset
# from sklearn.datasets import make_blobs
# x, y = make_blobs(n_samples=300, n_features=2, 
#                   centers=[[0., 0.], [0.5, 0.1]], 
#                   cluster_std=0.25, center_box=(-1., 1.))
# y = y.reshape(-1, 1).astype('float32')
# x_train, x_test, y_train, y_test = train_test_split(x, y)
# with open('data/blobs.pkl', 'wb') as f:
#     pickle.dump([x_train, x_test, y_train, y_test], f)

with open('data/blobs.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
    
# Visually see the distribution of the data points
plt.figure(figsize=(5, 5))
color = [['red', 'blue'][int(a)] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# Create an ANN model.
n_input = x_train.shape[1]  # number of input neurons
n_output = 1                # number of output neurons
n_hidden = 32               # number of hidden neurons
R = 0.001                   # Regularization constant
e = 1e-6                    # small value to avoid log(0)
adam = optimizers.Adam(learning_rate=0.001)

# The data is simple, but we intentionally added many hidden 
# layers to the model to demonstrate the effect of regularization.
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation = 'relu')(x_input)

# 4 more hidden layers
for i in range(4):
    h = Dense(n_hidden, activation = 'relu')(h)

y_output = Dense(n_output, activation='sigmoid')(h)
model = Model(x_input, y_output)
model.summary()

# 0: no regularization, 1: L1, 2: L2, 12: L1 & L2
params = model.trainable_variables
r_list = [1, 0, 2, 2, 12, 2, 1, 2, 0, 2]

# Custom loss function: regularized loss
class reg_loss(tf.keras.losses.Loss):
    def __init__(self, reg_lambda, params, r_list, **kwargs):
        super().__init__(**kwargs)
        self.R = reg_lambda
        self.params = params
        self.r_list = r_list
    
    def call(self, y, y_pred):
        bce = -tf.reduce_mean(
                  y * tf.math.log(y_pred + e) + \
                  (1. - y) * tf.math.log(1. - y_pred + e))
        
        reg_terms = 0
        for i, p in zip(self.r_list, self.params):
            if i == 1:  # L1 regularization
                reg_terms += tf.reduce_sum(tf.math.abs(p))
                
            if i == 2:  # L2 regularization
                reg_terms += tf.reduce_sum(tf.square(p))
                
            if i == 12:  # L1 & L2 regularization
                reg_terms += tf.reduce_sum(tf.math.abs(p))
                reg_terms += tf.reduce_sum(tf.square(p))
                
        return bce + self.R * reg_terms

model.compile(loss=reg_loss(R, params, r_list), optimizer = adam)
f = model.fit(x_train, y_train,
              validation_data=[x_test, y_test],
              epochs=200, batch_size=20)

# Visually see the loss history
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:.2f}".format(acc))

