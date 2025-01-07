# [MXDL-13-01] 2.noise_reduction(2).py
# Visually examine the images the reduced images
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Wf8o_w1C0VM
#
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Read an MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)
x = x.reshape(-1, 28, 28, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)
n_filters = 10

# Build a CNN-autoencoder model.
x_input = Input(batch_shape=(None, *x_train.shape[1:]))

# Encoder
e_conv = ZeroPadding2D((1,1))(x_input)
e_conv = Conv2D(filters=n_filters,
                kernel_size=(3,3),
                strides=2, 
                activation='relu')(e_conv)

# Decoder
d_conv = Conv2DTranspose(filters=n_filters,
                         kernel_size=(3,3),
                         strides=2,
                         padding = 'same', 
                         activation='relu')(e_conv)
x_output = Conv2D(filters=1, 
                  kernel_size=(3,3), 
                  strides=1,
                  padding = 'same', 
                  activation='sigmoid')(d_conv)

model = Model(x_input, x_output)
encoder = Model(x_input, e_conv)
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(lr=0.005))
model.summary()

# Training
h = model.fit(x_train, x_train, epochs = 100, batch_size=300)

# Loss History
plt.plot(h.history['loss'])
plt.show()

# Visualize images
def show_image(x, n):
    plt.figure(figsize=(14, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        ax.imshow(x[i], cmap='gray')
        ax.axis('off')
    plt.show()
    
# Visualize the test images
n_img = 10
print("\nOriginal images (28, 28)", end='')
show_image(x_test, 10)

# Visualize the reduced representation of x_test
r_test = encoder.predict(x_test, verbose=0)
print("\nReduced images (14, 14)", end='')
show_image(r_test.mean(axis=3), 10)
