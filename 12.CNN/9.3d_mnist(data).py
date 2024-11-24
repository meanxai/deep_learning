# [MXDL-12-05] 9.3d_mnist(data).py
#
# Code source: https://github.com/doleron/augmented_3d_mnist/blob/main/mnist_3d.ipynb
# Data source: https://www.kaggle.com/datasets/doleron/augmented-mnist-3d
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/tpP97EfsXco
#
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

df = h5py.File('data/3d-mnist-luiz.h5', 'r')
x_train = np.array(df["train_x"])  # (60000, 16, 16, 16, 3)
x_test  = np.array(df["test_x"])   # (10000, 16, 16, 16, 3)
y_train = np.array(df["train_y"])  # (60000,)
y_test  = np.array(df["test_y"])   # (10000,)

def print_grid(grid, azim, elev):
    grid_shape = grid.shape
    flattened = grid.reshape(((grid_shape[0] * grid_shape[1] * grid_shape[2]), 3))
    voxel_grid_array = np.zeros(len(flattened))

    for i in range(len(flattened)):
        temp = flattened[i]
        if temp[0] > 0 or temp[1] > 0 or temp[2] > 0:
            voxel_grid_array[i] = 1

    voxel_grid = voxel_grid_array.reshape((grid_shape[0], grid_shape[1], grid_shape[2]))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.azim = azim
    ax.elev = elev
    ax.voxels(voxel_grid, facecolors=grid)
    plt.show()

# Visually examine one 3D image.
m = 7
print_grid(x_train[m], 20, 60)
print('Target =', y_train[m])

m = 16
print_grid(x_train[m], 20, 60)
print('Target =', y_train[m])


