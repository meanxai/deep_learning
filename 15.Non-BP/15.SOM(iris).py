# [MXDL-15-12] 15.SOM(iris).py
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pickle

# Load and normalize the Iris dataset
x, y = load_iris(return_X_y = True)
x_input = (x - x.mean(axis=0)) / x.std(axis=0)
y_target = y

# Initialize the weights between the input layer of size 4 
# and the competitive layer of size 50x50. w: (2500, 4)
m_rows, m_cols = (50, 50)
w = np.random.normal(0, 0.1, size=(m_rows * m_cols, x_input.shape[1]))

ALPHA = 0.05              # learning rate
d_neighbor = 20           # the range of neighbors
n_data = x_input.shape[0] # the number of data points

# Among the neurons in the competitive layer, find the winner 
# neuron with the shortest distance between the input vector and 
# the weights.
def find_winner(w, x):
    dist = np.sqrt(np.sum(np.square(x - w), axis=1))  # distance
    winner = np.argmin(dist)
    return dist[winner], winner

# Find the neighbors around the winner neuron
# winner: 1D index of the winner neuron
# k: the maximum distance between the winner and its neighbors
def find_neighbors(winner, k):
    # convert 1D index to 2D coordinates
    # ex: winner=751 --> (row, col)=(15, 1)
    row, col = (winner // m_rows, winner % m_cols)
    from_i = np.max([0, row - k])
    to_i = row + k + 1
    if to_i > m_rows - 1:
        to_i = m_rows
    
    from_j = np.max([0, col - k])
    to_j = col + k + 1
    if to_j > m_cols - 1:
        to_j = m_cols
    
    neighbors = [i * m_cols + j for i in range(from_i, to_i)\
                                for j in range(from_j, to_j)]
    return np.array(neighbors)

# Update the weights using the competitive learning rule.
# Only update the weights connected to the winner and neighbors.
# Learning rule : w = w + alpha * (x - w)
def update_weights(w, winner, x):
    w[winner, :] += ALPHA * (x.reshape(-1,) - w[winner, :])
    return w

# 1. Training
err = []
for i in range(300):
    err.append(0)
    n_sample = 100
    for j in np.random.choice(n_data, n_sample):
        dx = x_input[j, :].reshape(1, -1)
        
        # Find the winner neuron
        dist, winner = find_winner(w, dx)
        
        # Find the neighbors of the winner neuron
        neighbors = find_neighbors(winner, d_neighbor)
        
        # Update the weights connected to the neighbors 
        for m in neighbors:
            w = update_weights(w, m, dx)
            
        err[-1] += dist / n_sample
    
    # Reduce the maximum distance between the winner and 
    # its neighbors by 1.
    if i % 3 == 0:
        d_neighbor = np.max([0, d_neighbor - 1])
            
    print("{} error = {:.4f}, d_neighbor = {}"\
          .format(i+1, err[-1], d_neighbor))

# Plot the error history
plt.plot(err)
plt.show()

# 2. Mapping
# Find the 2D coordinates of the winner neurons for given 
# data points in the competitive layer.
winners = []
for n in range(x_input.shape[0]):
    x = x_input[n].reshape(1, -1)
    _, winner = find_winner(w, x)
    
    row = winner // m_rows
    col = winner % m_cols
    winners.append((row, col))
winners = np.array(winners)

# Visualize the self-organizing map.
plt.figure(figsize=(6,6))
mark = ['o', 's', '^']
for i in np.unique(y_target):
    idx = np.where(y_target == i)[0]
    z = winners[idx]
    plt.scatter(z[:, 0], z[:, 1], s=300, marker=mark[i],
                alpha=0.7)
plt.legend(['setosa', 'versicolor', 'virginica'],
           bbox_to_anchor=(1, 1), prop={'size': 15})
plt.title('A Self-Organizing Map for Iris dataset')
plt.show()
