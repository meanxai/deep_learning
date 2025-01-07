# [MXDL-13-02] 4.denoise_mnist.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Q8AyijWiJyk
#
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Load a noised fashion MNIST dataset
# with open('data/noised_mnist.pkl', 'rb') as f:
#         x, y = pickle.load(f)

# Load a blurred fashion MNIST dataset
with open('data/blurred_mnist.pkl', 'rb') as f:
        x, y = pickle.load(f)
        
x_train, x_test, y_train, y_test = train_test_split(x, y)

# CNN AutoEncoder.
# Encoder
x_input = Input(batch_shape=(None, *x_train.shape[1:]))
e_conv = Conv2D(10, (3,3), strides=1, 
                padding = 'same', activation='relu')(x_input)

# Decoder
d_conv = Conv2DTranspose(10, (3,3), strides=1, 
                padding = 'same', activation='relu')(e_conv)
x_output = Conv2D(filters=1, kernel_size=(3,3), strides=1, 
                padding = 'same', activation='sigmoid')(d_conv)

model = Model(x_input, x_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005))
model.summary()

# Tranining
h = model.fit(x_train, y_train, epochs = 100, batch_size=300,
              validation_data=[x_test, y_test])

# Loss History
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.show()

# Denoising x_test
x_pred = model.predict(x_test, verbose=0)

# Visualize x_test and x_pred to see how much noise has been removed.
n = 5
idx = np.random.choice(x_test.shape[0], n)
fig, ax = plt.subplots(n, 3, figsize=(5, 8))

for i, k in enumerate(idx):
    ax[i, 0].imshow(x_test[k], cmap='gray')
    ax[i, 1].imshow(x_pred[k], cmap='gray')
    ax[i, 2].imshow(y_test[k], cmap='gray')
    ax[i, 0].axis('off')
    ax[i, 1].axis('off')
    ax[i, 2].axis('off')

    if i == 0:
        ax[i, 0].set_title("Blurred images")
        ax[i, 1].set_title("Deblurred images")
        ax[i, 2].set_title("Clear images")
plt.show()


