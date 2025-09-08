# [MXDL-15-04] 5.Ising_model.py
# Ising Model simulation
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import colors
cmap = colors.ListedColormap(['blue', 'yellow'])

k = 100        # size of the square lattice, k * k
epochs = 1000
J = 1.0        # coupling strength between neighbor spins
kT = 2.2       # critical temperature
lattice = np.random.choice([1,-1], [k, k])  # (-1 or +1)

@nb.jit(nopython=True)
def calculate_energy(sites):
    energy = 0
    for x in range(sites.shape[0]):
        for y in range(sites.shape[1]):
            s = sites[x, y]
            sum_spins = sites[(x + 1) % k, y] + \
                        sites[x, (y+1) % k] + \
                        sites[(x-1) % k, y] + \
                        sites[x,(y-1) % k]
            energy -= sum_spins * s
    return energy

# Ising Model simulation by metropolis algorithm
# reference: Ashkan Shekaari, MahmoudJafari, 2021,
#            Theory and simulation of the Ising model.
#            function Metropolis in 3.4.1 code1.py
@nb.jit(nopython=True)
def change_state(sites, n_iters):
    for i in range(n_iters):
        # Randomly choose a lattice site
        n = np.random.randint(k)
        m = np.random.randint(k)
        s = sites[n, m]

        # The sum of the spins of the nearest neighbors around s.
        sum_spins = sites[(n + 1) % k, m] + \
                    sites[n, (m+1) % k] + \
                    sites[(n-1) % k, m] + \
                    sites[n,(m-1) % k]
        
        # delta E = difference before and after flipping the spin s.
        dE = -J * (sum_spins * (-s) - sum_spins * s)

        if dE < 0:
            sites[n, m] *= -1            # flip the spin s
        else:
            if np.random.random() < np.exp(-dE / kT):
                sites[n, m] *= -1        # flip the spin s

    return sites

energy = []
for t in range(epochs):
    lattice = change_state(lattice, k * k)
    energy.append(calculate_energy(lattice))

    # plot the spins
    if t % 100 == 0:
        pos = str((lattice > 0).sum())
        neg = str((lattice < 0).sum())
        plt.figure(figsize=(4,4))
        plt.imshow(lattice, cmap=cmap)
        plt.title("iteration= "+str(t)+", (+1)="+pos+", (-1)="+neg)
        plt.show()
        
plt.plot(energy)
plt.show()
