# [MXDL-15-04] 6.Hopfield(2).py
import numpy as np
import matplotlib.pyplot as plt

# Input data: shape = (3, 9)
s = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 0],  # the letter 'T'
              [1, 0, 1, 1, 0, 1, 1, 1, 1],  # the letter 'U'
              [1, 1, 1, 1, 0, 0, 1, 1, 1]]) # the letter 'C'
s = 2 * s -1  # bipolar state
n, m = s.shape

w = s.T @ s   # weight matrix
np.fill_diagonal(w, 0)
print(w.round(2))

sp = np.array([[ 1, 0, 1, 0, 1, 0, 0, 1, 0],    # corrupted 'T'
               [ 0, 1, 1, 1, 0, 0, 0, 1, 1],    # corrupted 'C'
               [ 1, 0, 1, 0, 0, 1, 1, 1, 0]])   # corrupted 'U'
sp = 2 * sp -1  # bipolar state

def energy(s, w):
    return -0.5 * np.dot(np.dot(s.T, w), s)

for p in sp:
    fig, ax = plt.subplots(1, 3, figsize=(10,2))
    ax[0].imshow(p.reshape(3,3))
    ax[0].set_title('corrupted image')
    
    energies = [energy(p, w)]
    for k in range(50):
        E1 = energy(p, w) # Energy before flip
        
        i = np.random.choice(m, 1)
        p[i] *= -1        # flip the spin
        E2 = energy(p, w) # Energy after flip
        
        # delta E = difference before and after flipping the spin.
        dE = E2 - E1
        
        if dE >= 0:
            p[i] *= -1 # flip the spin back to its original state

        E = energy(p, w)
        energies.append(E)
    
    ax[1].imshow(p.reshape(3,3))
    ax[1].set_title('reconstructed image')
    
    ax[2].plot(energies, color='red', linewidth=3)
    ax[2].set_title('energy')
    plt.show()

