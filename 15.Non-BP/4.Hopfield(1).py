# [MXDL-15-03] 4.Hopfield(1).py
import numpy as np
import matplotlib.pyplot as plt

# Input data: shape = (3, 9)
s = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 0],  # the letter 'T'
              [1, 0, 1, 1, 0, 1, 1, 1, 1]]) # the letter 'U'
n, m = s.shape

w = (2*s-1).T @ (2*s-1)
np.fill_diagonal(w, 0)
print(w.round(2))

sp = np.array([[ 1, 0, 1, 0, 1, 0, 0, 1, 0],  # corrupted 'T'
               [ 1, 0, 1, 0, 0, 1, 1, 1, 0]]) # corrupted 'U'

for p in sp:
    fig, ax = plt.subplots(1, 2, figsize=(8,2.5))
    ax[0].imshow(p.reshape(3,3))
    ax[0].set_title('corrupted image')
    
    for k in range(10):
        i_rnd = np.random.choice(m, (m,), replace=False)
        for i in i_rnd:
            activation = w[i, :] @ p
            if activation > 0:
                p[i] = 1
            elif activation == 0:
                ;
            else:
                p[i] = 0
    
    ax[1].imshow(p.reshape(3,3))
    ax[1].set_title('reconstructed image')
    plt.show()

