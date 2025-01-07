# [MXDL-13-02] 3.noisy_blur_mnist.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Q8AyijWiJyk
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2

# Load a fashion MNIST dataset
f_mnist = tf.keras.datasets.fashion_mnist
(x, y), (x_test, y_test) = f_mnist.load_data()
x = x / 255

# Add noise to x
x_noise = x + 0.3 * np.random.normal(size=x.shape)
x_noise = np.clip(x_noise, 0., 1.)
x_noise = x_noise.reshape(-1, 28, 28, 1)
x_orig = x.reshape(-1, 28, 28, 1)

# Visualize images   
def show_image(x1, x2, n):
    idx = np.random.choice(x1.shape[0], n)
    fig, ax = plt.subplots(2, n, figsize=(10,4))
    for i in range(n):
        ax[0, i].imshow(x1[idx][i], cmap='gray')
        ax[0, i].axis('off')
        
        ax[1, i].imshow(x2[idx][i], cmap='gray')
        ax[1, i].axis('off')
    plt.show()

print("\nNoised fashion MNIST images", end='')
show_image(x_orig, x_noise, 5)

# Save the noised fashion MNIST images
with open('data/noised_mnist.pkl', 'wb') as f:
	pickle.dump([x_noise, x_orig], f)

# Apply circular motion blur to fashion MNIST images.
kernel = cv2.circle(np.zeros((10, 10)),
                    center = (5,5),
                    radius = 3,
                    color = (255,),
                    thickness = -1)
x_blur = [cv2.filter2D(img, -1, kernel) for img in x]
x_blur = np.array(x_blur)
x_blur = np.expand_dims(x_blur, -1)

print("\nBlurred fashion MNIST images", end='')
show_image(x_orig, x_blur, 5)

# Save the blurred fashion MNIST images
with open('data/blurred_mnist.pkl', 'wb') as f:
	pickle.dump([x_blur, x_orig], f)
