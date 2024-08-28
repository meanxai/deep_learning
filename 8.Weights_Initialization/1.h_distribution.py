# [MXDL-8-01] 1.h_distribution.py
# Observe the distribution of hidden layer outputs according 
# to initial weights
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/9I6CsvMqUPA
#
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Generate a simple dataset with 1000 data points
x = np.random.normal(size=(1000, 100))

# Create an ANN model with 2 hidden layers
n_input = x.shape[1]
n_hidden = 100      

# Let's check the distribution of hidden layer 
# outputs by changing std in N(0, std).
std = [1.0, 0.3, 0.15, 0.1, 0.05, 0.01]
for sigma in std:
    # Create an ANN model
    w1 = initializers.RandomNormal(mean=0.0, stddev=sigma)
    w2 = initializers.RandomNormal(mean=0.0, stddev=sigma)
    x_input = Input(batch_shape=(None, n_input))
    s1 = Dense(n_hidden, kernel_initializer = w1)(x_input)
    s2 = Dense(n_hidden, kernel_initializer = w2)(s1)
    
    s1_model = Model(x_input, s1)
    s2_model = Model(x_input, s2)
    
    i = 0
    s1_out = s1_model.predict(x, verbose=0)[:, i]
    s2_out = s2_model.predict(x, verbose=0)[:, i]
             
    # Check the distribution of the hidden layer outputs.
    plt.figure(figsize=(8,2))
    plt.subplot(121)
    plt.hist(s1_out, bins=30, color='blue', alpha=0.5)
    plt.title('w_std=' + str(sigma) + ', \
               s1_std=' + str(s1_out.std().round(3)))
    
    plt.subplot(122)
    plt.hist(s2_out, bins=30, color='red', alpha=0.5)
    plt.title('s2_std=' + str(s2_out.std().round(3)))
    plt.show()
