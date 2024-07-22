# [MXDL-4-01] 1.tf_binary_class.py
# Binary classification
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/1lc2A4jJSDE
#
import numpy as np
import tensorflow as tf
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
lr = 0.05             # learning rate

# Initialize the parameters
wh = tf.Variable(np.random.normal(size=(n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape=(1, n_hidden)))
wo = tf.Variable(np.random.normal(size=(n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape=(1, n_output)))
parameters = [wh, bh, wo, bo]
    
# loss function
def binary_crossentropy(y, y_hat):
    return -tf.reduce_mean(y * tf.math.log(y_hat) + (1. - y) * tf.math.log(1. - y_hat))

def predict(x, proba=True):
    p = parameters
    o_hidden = tf.nn.relu(tf.matmul(x, p[0]) + p[1])
    o_output = tf.sigmoid(tf.matmul(o_hidden, p[2]) + p[3])
    
    if proba:
        return o_output   # return sigmoid output as is
    else: 
        return (o_output.numpy() > 0.5) * 1  # return class

def fit(x_trn, y_trn, x_val, y_val, epochs, batch_size):
    trn_loss = []
    val_loss = []
    for epoch in range(epochs):
        # Training with mini-batch
        for batch in range(int(x_trn.shape[0] / batch_size)):
            idx = np.random.choice(x_trn.shape[0], batch_size)
            x_bat = x_trn[idx]
            y_bat = y_trn[idx]
            
            # Automatic differentiation
            with tf.GradientTape() as tape:
                loss = binary_crossentropy(y_bat, predict(x_bat))
                
            # Find the gradients of loss w.r.t the parameters
            grads = tape.gradient(loss, parameters)
                
            # update parameters by the gradient descent
            for i, p in enumerate(parameters):
                p.assign_sub(lr * grads[i])  # p = p - lr * gradient
        
        # loss history
        loss = binary_crossentropy(y_trn, predict(x_trn))
        trn_loss.append(loss.numpy())
        
        loss = binary_crossentropy(y_val, predict(x_val))
        val_loss.append(loss.numpy())
        
        if epoch % 10 == 0:
            print("{}: train_loss={:.4f}, val_loss={:.4f}".\
                  format(epoch, trn_loss[-1], val_loss[-1]))
                
    return trn_loss, val_loss

# training        
trn_loss, val_loss = fit(x_train, y_train, x_test, y_test, 
                           epochs=200, batch_size=50)

# Visually see the loss history
plt.plot(trn_loss, c='blue', label='train loss')
plt.plot(val_loss, c='red', label='validation loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = predict(x_test, proba=False)
acc = (y_pred == y_test).mean()
print("\nAccuracy of the test data = {:4f}".format(acc))