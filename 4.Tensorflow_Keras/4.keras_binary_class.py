# [MXDL-4-02] 4.keras_binary_class.py
# Binary classification
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/oemrJonU-tE
#
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a data set
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.1]], 
                  cluster_std=0.2, center_box=(-1., 1.))
y = y.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Visually see the data.
plt.figure(figsize=(6,4))
color = [['red', 'blue'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# Create an ANN with a hidden layer
n_input = x.shape[1]  # number of input neurons
n_output = 1          # number of output neurons
n_hidden = 8          # number of hidden neurons
adam = optimizers.Adam(learning_rate=0.01)

# Create an ANN model
model = Sequential()
model.add(Dense(n_hidden, input_dim=n_input, activation='relu'))
model.add(Dense(n_output, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam)

# training       
f = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              epochs=200, batch_size=50)

# Visually see the loss history
plt.plot(f.history['loss'], c='blue', label='train loss')
plt.plot(f.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:.4f}".format(acc))

