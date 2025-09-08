# [MXDL-15-05] 7.Hopfield(hebbian).py
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load MNIST data set
with open('data/mnist.pkl', 'rb') as f:
    x, y = pickle.load(f)     # x: (70000, 784), y: (70000, 1)

x = x[:3]
x = np.where(x > 0, 1, -1)
n, m = x.shape

# Hebbian learning rule
w = (1/m) * np.dot(x.T, x)   # weight matrix
np.fill_diagonal(w, 0)

def energy(s, w):
    E = -0.5 * np.dot(np.dot(s.T, w), s)
    return E

def corrupted_img(p):
    p.reshape(28,28)[18:, :] = -1
    return p.reshape(-1,)

for t in range(n):
    xh = corrupted_img(x[t])
    
    fig, ax = plt.subplots(1, 3, figsize=(9, 2))
    ax[0].imshow(xh.reshape(28, 28))
    ax[0].axis('off')
    ax[0].set_title('corrupted image')
    
    energies = [energy(xh, w)]
    for k in range(100):
        i_rnd = np.random.choice(m, (50,), replace=False)
        for i in i_rnd:
            E1 = energy(xh, w)  # Energy before flip
            
            xh[i] *= -1         # flip the spin
            E2 = energy(xh, w)  # Energy after flip
            
            # delta E = difference before and after flipping the spin.
            dE = E2 - E1
            
            if dE < 0:
                ;              # xh[i] is already flipped above
            else:
                xh[i] *= -1    # flip back to the original spin
            
            E = energy(xh, w)
            energies.append(E)
                   
    ax[1].imshow(xh.reshape(28,28))
    ax[1].axis('off')
    ax[1].set_title('reconstructed image')
    
    ax[2].plot(energies, color='red', linewidth=4)
    ax[2].set_title('energy')
    plt.show()

