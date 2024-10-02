# [MXDL-11-03] 4.compute_attention.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/k2iDqmp4Oeo
#
from tensorflow.keras.layers import Dot, Activation, Concatenate
import numpy as np

E = np.array([[[0.707, 0.616, 0.852],
               [0.19 , 0.113, 0.123],
               [0.757, 0.022, 0.236],
               [0.54 , 0.923, 0.412]]])

D = np.array([[[0.786, 0.634, 0.873],
               [0.796, 0.949, 0.872],
               [0.704, 0.314, 0.912],
               [0.293, 0.075, 0.73 ]]])

def AttentionLayer(d, e):
    dot_product = Dot(axes=(2, 2))([d, e])
    score = Activation('softmax')(dot_product)
    value = Dot(axes=(2, 1))([score, e])
    output = Concatenate()([value, d])
    return dot_product, score, value, output

d, s, v, o = AttentionLayer(D, E)

print("\nDot-product:")
print(np.round(d, 3))

print("\nScore:")
print(np.round(s, 3))

print("\nAttention values:")
print(np.round(v, 3))

print("\nAttentional hidden states:")
print(np.round(o, 3))

# verification of the first row of attention value
v1 = np.round(E[0,0,:] * s[0,0,0], 3)
v2 = np.round(E[0,1,:] * s[0,0,1], 3)
v3 = np.round(E[0,2,:] * s[0,0,2], 3)
v4 = np.round(E[0,3,:] * s[0,0,3], 3)
v=np.array([v1,v2,v3,v4])
np.round(np.sum(v, axis=0), 2)

