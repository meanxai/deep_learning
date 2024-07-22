# [MXDL-4-02] 6.tf_keras_regression.py
# Nonlinear Regression
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

# loss function: mean squared error
def mse(y, y_hat):
    return tf.reduce_mean(tf.math.square(y - y_hat))
    
# Create an ANN model
n_input = x.shape[1]  # number of input neurons
n_output = 1          # number of output neurons
n_hidden = 8          # number of hidden neurons
opt = optimizers.Adam(learning_rate=0.01)

x_input = Input(batch_shape=(None, n_input))
h_hidden = Dense(n_hidden, activation='tanh')(x_input)
y_output = Dense(n_output, activation='linear')(h_hidden)
model = Model(x_input, y_output)

# Update parameters using tf.GradientTape() and optimizer
def fit(x_trn, y_trn, x_val, y_val, epochs, batch_size):
    trn_loss = []
    val_loss = []
    for epoch in range(epochs):
        # Training with mini-batch
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x_trn.shape[0], batch_size)
            x_bat = x_trn[idx]
            y_bat = y_trn[idx]
            
            # Automatic differentiation
            with tf.GradientTape() as tape:
                loss = mse(y_bat, model(x_bat))
                
            # Find the gradients of loss w.r.t the parameters
            grads = tape.gradient(loss, model.trainable_variables)
                
            # update parameters by optimizer
            opt.apply_gradients(zip(grads, model.trainable_variables))
        
        # loss history
        loss = mse(y_trn, model(x_trn))
        trn_loss.append(loss.numpy())
        
        loss = mse(y_val, model(x_val))
        val_loss.append(loss.numpy())
        
        if epoch % 10 == 0:
            print("{}: train_loss={:.4f}, val_loss={:.4f}".\
                  format(epoch, trn_loss[-1], val_loss[-1]))
                
    return trn_loss, val_loss

# training        
trn_loss, val_loss = fit(x_train, y_train, x_test, y_test, 
                         epochs=200, batch_size=50)

# Visually see the data.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=50, c='blue', alpha=0.5, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.5, label='valid')
plt.legend()
plt.show()

# Visually see the loss history.
plt.plot(trn_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
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

