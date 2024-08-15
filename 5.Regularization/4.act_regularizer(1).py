# [MXDL-5-02] 4.act_regularizer(1).py
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
adam = optimizers.Adam(learning_rate=0.001)
reg_L1 = regularizers.L1(0.01)

# The data is simple, but we intentionally added many hidden 
# layers to the model to demonstrate the effect of regularization. 
x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation = 'relu', 
          activity_regularizer=reg_L1)(x_input)

# 4 more hidden layers
for i in range(4):
    h = Dense(n_hidden, activation = 'relu', 
              activity_regularizer=reg_L1)(h)

y_output = Dense(n_output, activation='sigmoid')(h)

model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', 
              optimizer = adam)

f = model.fit(x_train, y_train, epochs=200, batch_size=20,
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

# Visually see the distribution of the data points
plt.figure(figsize=(5, 5))
color = [['red', 'blue'][int(a)] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=100, c=color, alpha=0.3)
plt.show()
