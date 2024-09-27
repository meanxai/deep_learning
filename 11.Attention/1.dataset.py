# [MXDL-11-02] 1.dataset.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/KdQmVoEsBoE
#
import numpy as np
import pickle

# Generate a dataset consisting of two noisy sine curves
n = 5000   # the number of data points
s1 = np.sin(np.pi * 0.06 * np.arange(n)) + np.random.random(n)
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n)) + np.random.random(n)
data = np.vstack([s1, s2]).T

# Generate the training data for a Seq2Seq model.
t = 50     # time step
m = np.arange(0, n-2*t+1)
xi_enc = np.array([data[i:(i+t), :] for i in m])           # encoder input
xi_dec = np.array([data[(i+t-1):(i+2*t-1), :] for i in m]) # decoder input
xo_dec = np.array([data[(i+t):(i+2*t), :] for i in m])     # decoder output

# Save the training data for later use
with open('dataset.pkl', 'wb') as f:
 	pickle.dump([data, xi_enc, xi_dec, xo_dec], f)

print("\nThe shape of the dataset:", data.shape)
print("The shape of the encoder input:", xi_enc.shape)
print("The shape of the decoder input:", xi_dec.shape)
print("The shape of the decoder output:", xo_dec.shape)