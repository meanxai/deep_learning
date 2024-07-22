# [MXDL-2-03] 2.adadelta(ex).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/_ZQCt8Allv8
#
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

# initial point and the target point
w0 = np.array([2, 3], dtype='float32')  # initial point
wt = np.array([4, 2], dtype='float32')  # target point
w = tf.Variable(w0)                     # (w1, w2)

# create Adadelta optimizer
# Note that Adadelta tends to benefit from higher initial learning 
# rate values compared to other optimizers. To match the exact form 
# in the original paper, use 1.0.
opt = optimizers.Adadelta(learning_rate=1.0, rho = 0.9, epsilon=1e-4)

path = [w.numpy()]
for i in range(10):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(w - wt))
        dw = tape.gradient(loss, [w])
    
    opt.apply_gradients(zip(dw, [w]))
    path.append(w.numpy())
path = np.array(path)

print(path)
