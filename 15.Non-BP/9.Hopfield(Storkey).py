# [MXDL-15-06] 9.Hopfield(storkey).py
import numpy as np
import numba as nb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# Load the MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
    x, y = pickle.load(f)     # x: (70000, 784), y: (70000, 1)

x = x[:50]
x = np.where(x > 0, 1, -1)
n, m = x.shape   # n=50, m=784

# Compute the weights by Storkey learning rule.
@nb.jit(nopython=True)
def update_weights(s, w, nw):
    for k in range(nw):
        i = np.random.randint(m)
        j = np.random.randint(m)
        
        if i != j:     # if not diagonal
            # Calculate the local field at neurons i,j
            mask = np.ones(m)
            mask[i] = mask[j] = 0   # constraints: k != i, j
            hij = np.sum(w[i] * s * mask, axis=1)
            hji = np.sum(w[j] * s * mask, axis=1)
            
            # Increment weight for the pattern
            dw = s[:, i]*s[:, j] - s[:, i]*hji - hij*s[:, j]
            w[i][j] += np.sum(dw / m)
    return w

w = np.zeros([m, m])
n_iter = 5      # number of iterations
n_size = 10     # batch size of the input patterns
n_batch = int(x.shape[0] / n_size)
for e in range(n_iter):
    for i in tqdm(range(n_batch), desc="epochs-" + str(e)):
        i = np.random.choice(n, (n_size,), replace=False)
        w = update_weights(x[i], w, int(m * m / 2))

def energy(s, w):
    E = -0.5 * np.dot(np.dot(s, w), s)
    return E

def corrupted_img(p):
    p.reshape(28,28)[18:, :] = -1
    return p.reshape(-1,)

for t in range(10):    
    p = corrupted_img(x[t])
    
    fig, ax = plt.subplots(1, 3, figsize=(9, 2))
    ax[0].imshow(p.reshape(28, 28))
    ax[0].axis('off')
    ax[0].set_title('corrupted image')
    
    energies = [energy(p, w)]
    for k in range(200):
        i_rnd = np.random.choice(m, (50,), replace=False)
        for i in i_rnd:
            activation = np.dot(w[i, :], p)
            if activation > 0:
                p[i] = 1
            elif activation == 0:
                ;
            else:
                p[i] = -1

            E = energy(p, w)
            energies.append(E)
                   
    ax[1].imshow(p.reshape(28,28))
    ax[1].axis('off')
    ax[1].set_title('reconstructed image')
    
    ax[2].plot(energies, color='red')
    ax[2].set_title('energy')
    plt.show()
