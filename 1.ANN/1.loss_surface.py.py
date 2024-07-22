# [MXDL-1-03] 1.loss_surface.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/nqzS3dEvIQ0
#
import matplotlib.pyplot as plt
import numpy as np

x = (np.random.rand(50, 3) - 0.5) * 2.
y = (np.random.rand(50, 1) > 0.5) * 1.0

def sigmoid(x): return 1. / (1. + np.exp(-x))
def mse(y, y_hat): return np.mean((y - y_hat) ** 2)

Wh = np.random.rand(3, 20)
Wo = np.random.rand(20, 1)
 
# MSE loss function
def loss(w1, w2):
    Wh[0,0] = w1
    Wo[0,0] = w2
    h = np.tanh(np.dot(x, Wh))
    y_hat = sigmoid(np.dot(h, Wo))

    return mse(y, y_hat)

# w1, w2 = np.meshgrid(np.arange(-5, 5, .05), np.arange(-5, 5, .05))
w1, w2 = np.meshgrid(np.arange(-20, 20, .1), np.arange(-20, 20, .1))
zs = np.array([loss(a, b) for [a, b] in zip(np.ravel(w1), np.ravel(w2))])
z = zs.reshape(w1.shape)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, w2, z, alpha=0.7)

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('loss')
ax.azim = -30
ax.elev = 50
plt.show()


