# [MXDL-3-02] 1.num_diff.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/qHpnSGWVumE
#
import numpy as np

x = np.array([[1.0]])
y = np.array([[1.0]])
h = 1e-4

w0 = np.array([[0.5]])
w1 = np.array([[0.5, 0.5]])
w2 = np.array([[0.5], [0.5]])
parameters = [w0, w1, w2]

def sigmoid(x): return 1. / (1. + np.exp(-x))
def relu(x):    return np.maximum(0, x)
def bce(y, y_hat):
    return -np.mean(y * np.log(y_hat) + \
                           (1. - y) * np.log(1. - y_hat))

def predict(x):
    p = parameters
    h1 = relu(np.dot(x, p[0]))
    h2 = relu(np.dot(h1, p[1]))
    return sigmoid(np.dot(h2, p[2]))

p_gradients = []
for p in parameters:
    grad = np.zeros_like(p)
    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            p_org = p[row, col]
            p[row, col] = p_org + h
            L1 = bce(y, predict(x))
            
            p[row, col] = p_org - h
            L2 = bce(y, predict(x))
            grad[row, col] = (L1 - L2) / (2. * h)
            p[row, col] = p_org
    p_gradients.append(grad)
    
for i in range(len(parameters)):    
    parameters[i] -= 0.1 * p_gradients[i]
    
print(parameters[2])    
print(parameters[1])
print(parameters[0])
