# [MXDL-12-01] 2.pooling.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/ZkhHvPdbQnI
#
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)

w1 = np.random.normal(size=(7,7)) # filter for convolutional layer
w2 = np.empty(shape=(3,3))        # filter for pooling layer 

def cross_correlation(x, w):
    return np.sum(x * w)

def feat_map(x, w, s):
    rw, cw = w.shape
    rx, cx = x.shape
    rf = int((rx - rw) / s + 1)
    cf = int((cx - cw) / s + 1)
    feat = np.zeros(shape=(rf, cf))
    for i in range(rf):
        for j in range(cf):
            px = x[(i*s):(i*s+rw), (j*s):(j*s+cw)]            
            feat[i, j] = cross_correlation(px, w)
    return np.maximum(0, feat)  # ReLU

def pooling(x, w, s, method):
    rw, cw = w.shape
    rx, cx = x.shape
    rf = int((rx - rw) / s + 1)
    cf = int((cx - cw) / s + 1)
    feat = np.zeros(shape=(rf, cf))
    for i in range(rf):
        for j in range(cf):
            px = x[(i*s):(i*s+rw), (j*s):(j*s+cw)]
            if method == 'max': feat[i, j] = np.max(px)
            else: feat[i, j] = np.mean(px)
    return feat

def show_image(x, size, title):
    plt.figure(figsize=size)
    plt.imshow(x)
    plt.title(title)
    plt.show()
    
xi = x[13].reshape(28,28)
show_image(xi, (5,5), "Input image")

feat = feat_map(xi, w1, 1)
show_image(feat, (4,4), "Feature map")

max_pool = pooling(feat, w2, 1, 'max')
show_image(max_pool, (3,3), "Max-pooling")

avg_pool = pooling(feat, w2, 1, 'avg')
show_image(avg_pool, (3,3), "Average-pooling")





