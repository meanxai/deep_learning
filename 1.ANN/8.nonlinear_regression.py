# [MXDL-1-07] 8.nonlinear_regression.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/HmunSzYCKtg
#
# Create a two-layered ANN model and perform non-linear 
# regression using numerical differentiation and gradient 
# descent.
#
# To calculate the gradient accurately, you must use 
# automatic differentiation. However, here we use numerical 
# differentiation to approximate the gradient.
# Gradient descent with automatic differentiation will be 
# discussed in detail in the backpropagation part later.
import numpy as np
from gradient_descent import gradient_descent
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate training data set
x = np.random.random((1000, 1))
y = 2.0 * np.sin(2.0 * np.pi * x) + \
    np.random.normal(0.0, 0.8, (1000, 1))

# Generate training, validation, and test data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y)
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

# See the data visually.
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.3, label='valid')
plt.legend()
plt.show()

# Create a two-layered ANN model.
n_input = x.shape[1]       # number of input neurons
n_output = 1               # number of output neurons
n_hidden = 16              # number of hidden1 neurons
alpha = 0.01               # learning rate
h = 1e-4                   # constant for numerical differentiation
    
# initialize the parameters randomly.
wh = np.random.normal(size=(n_input, n_hidden))  # weights of hidden layer
bh = np.zeros(shape=(1, n_hidden))               # bias of hidden layer
wo = np.random.normal(size=(n_hidden, n_output)) # weights of output layer
bo = np.zeros(shape=(1, n_output))               # bias of output layer
parameters = [wh, bh, wo, bo]

# loss function: mean squared error
def loss(y, y_hat):
    return np.mean(np.square(y - y_hat))

# Output from the ANN model: prediction process
def predict(x):
    p = parameters
    h_out = np.tanh(np.dot(x, p[0]) + p[1])  # output from hidden layer
    o_out = np.dot(h_out, p[2]) + p[3]       # output from output layer
    return o_out

# Perform training and track the loss history.
def train(x, y, x_val, y_val, epochs, batch_size):
    ht_loss = []  # loss history of training data
    hv_loss = []  # loss history of validation data
    for epoch in range(epochs):
        # measure the losses during training
        ht_loss.append(loss(y, predict(x)))         # loss for training data
        hv_loss.append(loss(y_val, predict(x_val))) # loss for validation data

        # Perform training using mini-batch gradient descent
        for batch in range(int(x.shape[0] / batch_size)):
            idx = np.random.choice(x.shape[0], batch_size)
            gradient_descent(x[idx], y[idx], alpha, loss, predict, parameters)
            
        if epoch % 10 == 0:
            print("{}: train_loss={:.4f}, val_loss={:.4f}".\
                  format(epoch, ht_loss[-1], hv_loss[-1]))
    return ht_loss, hv_loss

# Perform training
train_loss, val_loss = train(x_train, y_train, x_valid, y_valid, 
                             epochs=1000, batch_size=50)

# Visually check the loss history.
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Visually check the prediction result.
y_pred = predict(x_test)
plt.figure(figsize=(7,5))
plt.scatter(x_train, y_train, s=20, c='blue', alpha=0.3, label='train')
plt.scatter(x_valid, y_valid, s=20, c='red', alpha=0.3, label='validation')
plt.scatter(x_test, y_pred, s=5, c='red', label='test')
plt.legend()
plt.show()

