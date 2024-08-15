# [MXDL-6-02] 3.dropout.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/OrxJmX4WHTA
#
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a dataset for binary classification
x, y = make_blobs(n_samples=200, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.1]], 
                  cluster_std=0.25, center_box=(-1., 1.))

# Visually see the data
plt.figure(figsize=(5,4))
color = [['red', 'blue'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# Split the dataset into the training and test data.
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Create a model.
n_input = x.shape[1]  # number of input neurons
n_output = 1          # number of output neurons
n_hidden = 64         # number of hidden neurons
d_rate = 0.5          # dropout rate    
adam = optimizers.Adam(learning_rate=0.01)

x_input = Input(batch_shape=(None, n_input))
h = Dense(n_hidden, activation = 'relu')(x_input)
h = Dropout(rate=d_rate)(h)

# Additional 3 hidden layers
# The data is simple, but we intentionally added many hidden 
# layers to the model to demonstrate the effect of regularization.
for i in range(3):
    h = Dense(n_hidden, activation='relu')(h)
    h = Dropout(rate=d_rate)(h)

y_output = Dense(n_output, activation='sigmoid')(h)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=adam)

# Training        
h = model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              epochs=100, batch_size=100)

# Visually see the loss history
plt.plot(h.history['loss'], c='blue', label='train loss')
plt.plot(h.history['val_loss'], c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data.
y_pred = (model.predict(x_test) > 0.5) * 1
acc = (y_pred.reshape(-1,) == y_test).mean()
print("\nAccuracy of the test data = {:4f}".format(acc))
