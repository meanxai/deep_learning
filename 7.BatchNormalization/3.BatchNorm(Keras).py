# [MXDL-7-02] 3.BatchNorm(Keras).py
# Verify that BatchNormalization prevents overfitting.
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/_qeKY0I32nk
#
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, \
     BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pickle

# Read a dataset
with open('data/blobs.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
    
# Visually see the data
plt.figure(figsize=(5,4))
color = [['red', 'blue'][a] for a in y_train.reshape(-1,)]
plt.scatter(x_train[:, 0], x_train[:, 1], s=20, c=color, alpha=0.3)
plt.show()

# Create an ANN model with Batch Normalization layer
n_input = x_train.shape[1] # number of input neurons
n_output = 1               # number of output neurons
n_hidden = 128             # number of hidden neurons
adam = optimizers.Adam(learning_rate=0.01)

x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, use_bias=False)(x_input)
h = BatchNormalization()(h)
h = Activation('relu')(h)

# Additional 10 hidden layers
# The data is simple, but we intentionally added many hidden 
# layers to the model to verity the effect of Batch Normalization.
for i in range(10):
    h = Dense(n_hidden, use_bias=False)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

y_output = Dense(n_output, activation='sigmoid')(h)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=adam)

# training        
h = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              epochs=100, batch_size=300, shuffle=True)

# Visually see the loss history
plt.plot(h.history['loss'], c='blue', label='train loss')
plt.plot(h.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nThe accuracy of the test data = {:4f}".format(acc))