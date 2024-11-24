# [MXDL-12-06] 11.moving_mnist.py
# Preprocessing the moving MNIST dataset
# data source: http://www.cs.toronto.edu/~nitish/unsupervised_video/
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/FzUAPtDgA_o
#
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the moving MNIST dataset
data = np.load('data/mnist_test_seq.npy') # (20, 10000, 64, 64)
data = np.swapaxes(data, 0, 1)            # (10000, 20, 64, 64)
data = data[:5000, ...].astype('float32') # (5000, 20, 64, 64)
data /= 255.                              # normalization (0 ~ 1)
data = np.expand_dims(data, axis=-1)      # (5000, 20, 64, 64, 1)

# Split the dataset into training and test data
n_train = int(0.9 * data.shape[0])  # the number of training data
idx = np.arange(data.shape[0])
np.random.shuffle(idx)
d_train = data[idx[: n_train]]
d_test = data[idx[n_train :]]

def create_xy(x):
	t = x.shape[1]  # the number of time steps
	tx = x[:, 0:(t - 1), :, :, :]
	ty = x[:, 1:t, :, :, :]
	return tx, ty

# Create input and output data to feed into the model.
x_train, y_train = create_xy(d_train) # (4500, 19, 64, 64, 1)
x_test, y_test = create_xy(d_test)    # ( 500, 19, 64, 64, 1)

# Visualize the images.
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
idx = np.random.choice(n_train, 1)[0]
for i, ax in enumerate(axes.flat):
	ax.imshow(np.squeeze(d_train[idx][i]), cmap="gray")
	ax.set_title(f"Frame {i + 1}")
	ax.axis("off")
plt.show()

# save the training and test data
with open('data/mv_mnist.pkl', 'wb') as f:
	pickle.dump([x_train, y_train, x_test, y_test, d_test], f)

