# [MXDL-2-03] 3.optimizers(tf).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/_ZQCt8Allv8
#
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as op
import matplotlib.pyplot as plt

# initial point and the target point
w0 = np.array([0.1, 3.5], dtype='float32')  # initial point
wt = np.array([4, 2], dtype='float32')  # target point
w = tf.Variable(w0)                     # (w1, w2)

opt = []
opt_name = ["SGD", "Momentum", "NAG", "Adagrad", 
            "RMSprop", "Adadelta", "Adam"]
opt.append(op.SGD(learning_rate = 0.8, momentum=0))
opt.append(op.SGD(learning_rate = 0.8, momentum=0.2))
opt.append(op.SGD(learning_rate = 0.1, momentum=0.2, nesterov=True))
opt.append(op.Adagrad(learning_rate = 0.8))
opt.append(op.RMSprop(learning_rate = 0.2, rho=0.9))
opt.append(op.Adadelta(learning_rate=5.0, rho=0.9, epsilon=1e-4))
opt.append(op.Adam(learning_rate = 0.1, beta_1=0.9, beta_2=0.9))

def loss(w):
    return tf.reduce_sum(tf.square(w - wt) * [0.1, 1.2])

for k in range(len(opt)):
    w.assign(w0)
    path = [w.numpy()]
    for i in range(70):
        # perform automatic differentiation
        with tf.GradientTape() as tape:
            dw = tape.gradient(loss(w), [w])
        
        # update w by gradient descent
        opt[k].apply_gradients(zip(dw, [w]))
        path.append(w.numpy())
    path = np.array(path)
    
    # Draw the loss surface and the path to the optimal point.
    x, y = path[:, 0], path[:, 1]
    m = 5
    t = 0.1
    w1, w2 = np.meshgrid(np.arange(0, m, t), np.arange(0, 4, t))
    zs = np.array([loss([a,b]).numpy() for [a, b] in zip(np.ravel(w1), np.ravel(w2))])
    z = zs.reshape(w1.shape)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the surface of the loss function
    ax.plot_surface(w1, w2, z, alpha=0.7)
    
    # Dwaw the path to the optimal point.
    L = np.array([loss([a, b]).numpy() for [a, b] in zip(x, y)])
    ax.plot(x, y, L, marker='o', color="r")
    
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('loss')
    ax.set_title(opt_name[k], fontsize= 20)
    ax.azim = -50
    ax.elev = 50
    plt.show()

