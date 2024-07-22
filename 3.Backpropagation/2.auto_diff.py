# [MXDL-3-02] 2.auto_diff.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/qHpnSGWVumE
#
import numpy as np
import tensorflow as tf

x = np.array([[1.0]])
y = np.array([[1.0]])

w0 = tf.Variable(np.array([[0.5]]))
w1 = tf.Variable(np.array([[0.5, 0.5]]))
w2 = tf.Variable(np.array([[0.5], [0.5]]))
parameters = [w0, w1, w2]

# loss function
def bce(y, y_hat):
    return -y * tf.math.log(y_hat) - \
                           (1. - y) * tf.math.log(1. - y_hat)

def predict(x):
    h1 = tf.nn.relu(tf.matmul(x, parameters[0]))
    h2 = tf.nn.relu(tf.matmul(h1, parameters[1]))
    return tf.sigmoid(tf.matmul(h2, parameters[2]))

print(predict(x))

with tf.GradientTape() as tape:
    loss = bce(y, predict(x))
grads = tape.gradient(loss, parameters)

for i, p in enumerate(parameters):
    p.assign_sub(0.1 * grads[i]) 

print(w2.numpy())
print(w1.numpy())
print(w0.numpy())
