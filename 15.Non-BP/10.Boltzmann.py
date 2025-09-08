# [MXDL-15-07] 10.Boltzmann.py
import numpy as np
import matplotlib.pyplot as plt

# Input data
s = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 0],  # 'T'
              [1, 0, 1, 1, 0, 1, 1, 1, 1],  # 'U'
              [1, 0, 0, 1, 0, 0, 1, 1, 1],  # 'L'
              [1, 0, 1, 1, 1, 1, 1, 0, 1]]) # 'H'
n,m = s.shape

# Compute the energy
def energy(x, w):
    return -0.5 * np.dot(np.dot(x, w), x)

# Initialize weights randomly
w = np.random.normal(size=(m, m))
np.fill_diagonal(w, 0)

# Update the weights by the learning rule
E = []
for epoch in range(1000):
    for k in range(100):
        i, j = np.random.randint(m, size=2)
        if i != j:
            # Compute the mean of si * sj from the data
            E_data = np.mean(s[:, i] * s[:, j])
            
            # Expectation of si * sj from the model
            y = np.random.randint(0, 2, size=(1000, m))
            dE = np.dot(y, w)                    # energy gap
            pk = 1 / (1 + np.exp(-dE))           # probability
            y = np.random.binomial(n=1, p=pk)    # sampling
            E_model = np.mean(y[:, i] * y[:, j])
            
            # Update the weights by gradient ascent
            w[i, j] += 0.1 * (E_data - E_model)
        
    if epoch % 50 == 0:
        # Compute the total energy
        E.append(np.sum([energy(s[e], w) for e in range(n)]))
        print("Epochs:{}, Energy = {:.2f}".format(epoch, E[-1]))

# Plot the energy changes
plt.plot(E)
plt.show()

# Test data. Corrupted images
cx = np.array([[1, 1, 1, 0, 1, 0, 0, 0, 0],  # corrupted 'T'
               [1, 0, 1, 0, 0, 1, 1, 1, 1],  # corrupted 'U'
               [1, 0, 0, 1, 0, 0, 0, 1, 1],  # corrupted 'L'
               [1, 0, 1, 0, 1, 1, 1, 0, 1]]) # corrupted 'H'

# Reconstructing the corrupted images
for p in cx:
    fig, ax = plt.subplots(1, 2, figsize=(5,1.5))
    ax[0].imshow(p.reshape(3,3))
    ax[0].set_title('corrupted image')
    
    dE = np.dot(p, w)
    pk = 1 / (1 + np.exp(-dE))
    y = np.where(pk >= 0.5, 1, 0)
        
    ax[1].imshow(y.reshape(3,3))
    ax[1].set_title('reconstructed image')
plt.show()
