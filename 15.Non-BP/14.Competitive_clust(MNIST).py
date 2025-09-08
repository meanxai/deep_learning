# [MXDL-15-11] 14.Competitive_clust(MNIST).py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pickle

# Load MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
    x_input, _ = pickle.load(f)     # x_input: (70000, 784)

n_input = 784
n_output = 10
ALPHA = 0.01
n = x_input.shape[0]

# Among the neurons in the competitive layer, find the winner 
# neuron with the shortest distance between the input vector and 
# the weights.
def find_winner(x, w):
    x = x.reshape(1, -1)
    d = np.sqrt(np.sum(np.square(x.T - w), axis=0))  # distance
    winner = np.argmin(d)
    return np.min(d), winner

# Update the weights using the competitive learning rule.
# Only update the weights connected to the winner neuron.
# Learning rule : w = w + alpha * (x - w)
def update_weights(w, winner, x):
    w[:, winner] += ALPHA * (x - w[:, winner])
    return w

# Initialize the weights
w = np.random.normal(0, 0.1, size=(n_input, n_output))

# Training
for i in range(10):
    err = 0
    for k in range(n):
        dx = x_input[k]
        
        # Find the winner neuron
        d, winner = find_winner(dx, w)

        # Update the weights connected to the winner neuron
        w = update_weights(w, winner, dx)
        
        # The distance between dx and the weights connected 
        # to the winner. This can be considered an error in 
        # unsupervised learning.
        err += d
        
    print("{} done. error = {:.4f}".format(i+1, err / n))
    
# Predict cluster index for each data point.
clust = []
for k in range(n):
    dx = x_input[k].reshape(1, -1)
    _,winner = find_winner(dx, w)
    clust.append(winner)
clust = np.array(clust)

# Visualize the clustering results
for k in np.unique(clust):
    # Find the images belonging to cluster k
    idx = np.where(clust == k)[0]
    images = x_input[idx]
    
    # Centroid image
    centroid = w[:, k]
    
    # Find 10 images near the centroid image.
    d = np.sqrt(np.sum((images - centroid)**2, axis=1))
    nearest = np.argsort(d)[:10]
    images = x_input[idx[nearest]]
    
    # plot the centroid image
    f = plt.figure(figsize=(10, 2))
    image = centroid.reshape(28, 28)
    ax = f.add_subplot(1, 11, 1)
    ax.imshow(image, cmap=plt.cm.bone)
    ax.grid(False)
    ax.set_title("C")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.tight_layout()
    
    for i in range(10):
        image = images[i].reshape(28,28)
        ax = f.add_subplot(1, 11, i+2)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()