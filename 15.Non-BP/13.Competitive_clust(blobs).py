# [MXDL-15-11] 13.Competitive_clust(blobs).py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.pyplot import cm

# Generate a dataset with 3 Gaussian clusters
n = 1500
x_input, _ = make_blobs(n_samples = n, n_features = 2, 
                        centers=[(0., 0), (0.5, 0.5), (1., 0.)], 
                        cluster_std = 0.15)

n_input = 2     # the number of features
n_output = 3    # 3 clusters
ALPHA = 0.01    # learning rate

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
w = np.random.rand(n_input, n_output)

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
    _, winner = find_winner(dx, w)
    clust.append(winner)

# Visualize the clustering results
clust = np.array(clust)
color = cm.rainbow(np.linspace(0, 1, n_output))
plt.figure(figsize=(8, 6))
for i, c in zip(range(n_output),color):
    plt.scatter(x_input[clust == i, 0], x_input[clust == i, 1],
                s=20, color=c, marker='o', alpha=0.5, 
                label='cluster ' + str(i))
plt.scatter(w[0, :], w[1, :], s=250, marker='*', color='black', 
            label='centroids')
plt.title("Clustering results by competitive learning")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

