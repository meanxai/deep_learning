# [MXDL-4-01] 2.tf_multi_class.py
# Multiclass classification
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

# Generate a dataset for multiclass classification
x, y = make_blobs(n_samples=400, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.1], [1., 0.]], 
                  cluster_std=0.15, center_box=(-1., 1.))
n_class = np.unique(y).shape[0]  # the number of classes

# one-hot encode class y, y = [0,1,2]
y_ohe = np.eye(n_class)[y]

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe)

# Visually see the data.
plt.figure(figsize=(5,4))
color = [['red', 'blue', 'green'][a] for a in y.reshape(-1,)]
plt.scatter(x[:, 0], x[:, 1], s=100, c=color, alpha=0.3)
plt.show()

# Create an ANN with a hidden layer
n_input = x.shape[1]       # number of input neurons
n_output = n_class         # number of output neurons
n_hidden = 8               # number of hidden neurons
lr = 0.05                  # learning rate

# Initialize the parameters
wh = tf.Variable(np.random.normal(size=(n_input, n_hidden)))
bh = tf.Variable(np.zeros(shape=(1, n_hidden)))
wo = tf.Variable(np.random.normal(size=(n_hidden, n_output)))
bo = tf.Variable(np.zeros(shape=(1, n_output)))
parameters = [wh, bh, wo, bo]
opt = optimizers.Adam(learning_rate = 0.01, beta_1=0.9, beta_2=0.999)

# loss function
def crossentropy(y, y_hat):
    ce = -tf.reduce_sum(y * tf.math.log(y_hat), axis=1)
    return tf.reduce_mean(ce)

def predict(x, proba=True):
    p = parameters
    o_hidden = tf.nn.relu(tf.matmul(x, p[0]) + p[1])
    o_output = tf.nn.softmax(tf.matmul(o_hidden, p[2]) + p[3])
    
    if proba:
        return o_output   # return softmax output as is
    else: 
        return tf.math.argmax(o_output, axis=1)  # return class

def fit(x_trn, y_trn, x_val, y_val, epochs, batch_size):
    trn_loss, val_loss = [], []
    for epoch in range(epochs):
        # Training with mini-batch
        for batch in range(int(x_trn.shape[0] / batch_size)):
            idx = np.random.choice(x_trn.shape[0], batch_size)
            x_bat = x_trn[idx]
            y_bat = y_trn[idx]
            
            # Automatic differentiation
            with tf.GradientTape() as tape:
                loss = crossentropy(y_bat, predict(x_bat))
                
            # Find the gradients of loss w.r.t the parameters
            grads = tape.gradient(loss, parameters)
                
            # update parameters by the gradient descent
            opt.apply_gradients(zip(grads, parameters))
        
        # loss history
        loss = crossentropy(y_trn, predict(x_trn))
        trn_loss.append(loss.numpy())
        
        loss = crossentropy(y_val, predict(x_val))
        val_loss.append(loss.numpy())
        
        if epoch % 10 == 0:
            print("{}: train_loss={:.4f}, val_loss={:.4f}".\
                  format(epoch, trn_loss[-1], val_loss[-1]))
                
    return trn_loss, val_loss

# training        
train_loss, test_loss = fit(x_train, y_train, x_test, y_test, 
                           epochs=200, batch_size=50)

# Visually see the loss history
plt.plot(train_loss, c='blue', label='train loss')
plt.plot(test_loss, c='red', label='test loss')
plt.legend()
plt.show()

# Check the accuracy of the test data
y_pred = predict(x_test, proba=False).numpy()
acc = (y_pred == np.argmax(y_test, axis=1)).mean()
print("Accuracy of test data = {:4f}".format(acc))
