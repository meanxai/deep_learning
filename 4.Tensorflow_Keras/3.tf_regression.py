# [MXDL-4-01] 3.tf_regression.py
# Nonlinear Regression
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/1lc2A4jJSDE
#
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + \
    np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, test data set
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_pred = np.linspace(0, 1, 200).reshape(-1, 1)

# Visually see the data.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=50, c='blue', alpha=0.5, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.5, label='valid')
plt.legend()
plt.show()

# Create an ANN with a hidden layer
n_input = x.shape[1]  # number of input neurons
n_output = 1          # number of output neurons
n_hidden = 8          # number of hidden neurons
lr = 0.05             # learning rate

# Initialize the parameters
wh = tf.Variable(np.random.normal(size=(n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape=(1, n_hidden)))
wo = tf.Variable(np.random.normal(size=(n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape=(1, n_output)))
parameters = [wh, bh, wo, bo]
opt = optimizers.Adam(learning_rate = 0.01)

# loss function: mean squared error
def mse(y, y_hat):
    return tf.reduce_mean(tf.math.square(y - y_hat))
    
def predict(x, proba=True):
    p = parameters
    o_hidden = tf.math.tanh(tf.matmul(x, p[0]) + p[1])
    o_output = tf.matmul(o_hidden, p[2]) + p[3]
    return o_output

def fit(x_trn, y_trn, x_val, y_val, epochs, batch_size):
    trn_loss = []
    val_loss = []
    for epoch in range(epochs):
        # Training with mini-batch
        for batch in range(int(x_trn.shape[0] / batch_size)):
            idx = np.random.choice(x_trn.shape[0], batch_size)
            x_bat = x_trn[idx]
            y_bat = y_trn[idx]
            
            # Automatic differentiation and update parameters
            loss = lambda: mse(y_bat, predict(x_bat))
            opt.minimize(loss, parameters)
        
        # loss history
        loss = mse(y_trn, predict(x_trn))
        trn_loss.append(loss.numpy())
        
        loss = mse(y_val, predict(x_val))
        val_loss.append(loss.numpy())
        
        if epoch % 10 == 0:
            print("{}: train_loss={:.4f}, val_loss={:.4f}".\
                  format(epoch, trn_loss[-1], val_loss[-1]))
                
    return trn_loss, val_loss

# training       
trn_loss, val_loss = fit(x_train, y_train, x_test, y_test, 
                           epochs=200, batch_size=50)

# Visually see the loss history.
plt.plot(trn_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Visually check the prediction result.
y_pred = predict(x_pred)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_test, y_test, s=20, c='red', alpha=0.3, label='validation')
plt.scatter(x_pred, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()
