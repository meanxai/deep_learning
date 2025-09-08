# [MXDL-15-12] 16.SOM(mnist).py
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load and normalize the MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)

x_input = x[:2000] * 2 - 1   # -1 ~ +1
y_target = y.reshape(-1,)[:2000]

# Initialize the weights between the input layer of size 784 
# and the competitive layer of size 100x100. w: (10000, 784)
m_rows, m_cols = (100, 100)
w = np.random.normal(0, 0.1, size=(m_rows * m_cols, x_input.shape[1]))

ALPHA = 0.05                # learning rate
d_neighbor = 20             # the initial maximum distance 
                            # between the winner and its neighbors
n_data = x_input.shape[0]   # the number of data points

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
    n_sample = 500
    for j in np.random.choice(n_data, n_sample):
        dx = x_input[j, :].reshape(1, -1)
        
        # Find the winner neuron
        dist, winner = find_winner(w, dx)
        
        # Find the neighbors of the winner neuron
        if d_neighbor > 0:
            neighbors = find_neighbors(winner, d_neighbor)
        else:
            neighbors = [winner]
        
        # Update the weights connected to the winner neuron 
        # and its neighbors via Hebb's rule.
        for m in neighbors:
            w = update_weights(w, m, dx)
            
        err[-1] += dist / n_sample
    
    # Reduce the maximum distance between the winner and 
    # its neighbors by 1.
    if i % 2 == 0:
        d_neighbor = np.max([0, d_neighbor - 1])
            
    print("{} error = {:.2f}, d_neighbor = {}"\
          .format(i+1, err[-1], d_neighbor))

# Plot the error history
plt.plot(err)
plt.show()

# 2. Mapping
# Find the 2D coordinates of the winner neurons for given 
# data points in the competitive layer.
winners = []
for n in range(n_data):
    x = x_input[n].reshape(1, -1)
    _, winner = find_winner(w, x)
    
    row = winner // m_rows
    col = winner % m_cols
    winners.append((row, col))

winners = np.array(winners)

# Visualize the self-organizing map.
plt.figure(figsize=(8,8))
for i in np.unique(y_target):
    idx = np.where(y_target == i)[0]
    z = winners[idx]
    plt.scatter(z[:, 0], z[:, 1], s=150, alpha=0.7, label=str(i))
plt.legend(framealpha=1.0)
plt.title('A self organizing map for MNIST digits')
plt.show()


