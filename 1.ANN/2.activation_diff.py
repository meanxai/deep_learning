# [MXDL-1-03] 2.activation_diff.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/nqzS3dEvIQ0
#
import numpy as np
import matplotlib.pyplot as plt

# activation function
def sigmoid(x): return 1. / (1. + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def softplus(x): return np.log(1 + np.exp(x))

# Numerical Differentiation
def num_differentiation(f, x, h):
    return (f(x+h) - f(x-h)) / (2. * h)

x = np.linspace(-5, 5, 100)

h = 1e-08
#f = sigmoid
#f = relu
#f = tanh
f = softplus
fx = f(x)
gx = num_differentiation(f, x, h)

# Visualization
plt.figure(figsize=(7,6))
plt.plot(x, fx, label='function f(x)')
plt.plot(x, gx, label='derivatives g(x)')
plt.axvline(x=0, ls='--', lw=0.5, c='gray')
plt.legend()
plt.show()
