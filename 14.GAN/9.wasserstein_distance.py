# [MXDL-14-08] 9.wasserstein_distance.py
# Computing Wasserstein distance
# Source: https://visualstudiomagazine.com/articles
#               /2021/08/16/wasserstein-distance.aspx
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Cbg83Q3on-M
#
import numpy as np
from scipy.stats import wasserstein_distance

# Find the position of the first element in vector x 
# that is greater than 0. If all elements are 0, -1 is returned.
def first_nonzero(x):
    idx = np.where(x > 0)[0]
    if len(idx) == 0:
        return -1
    else:
        return idx[0]

# Move all or part of the dirt at i_from to the hole at i_to.
def move_dirt(x, y, dirt, i_from, holes, i_to):
    if dirt[i_from] <= holes[i_to]:
        flow = dirt[i_from]
        dirt[i_from] = 0.0
        holes[i_to] -= flow
    else:
        flow = holes[i_to]
        dirt[i_from] -= flow
        holes[i_to] = 0.0
    dist = np.abs(x[i_from] - y[i_to])
    return flow, dist, dirt, holes

# Compute Wasserstein distance
def my_wasserstein(x, y, dirt, holes):
    tot_work = 0.0
    while True:
        i_from = first_nonzero(dirt)
        i_to = first_nonzero(holes)
        if i_from == -1 or i_to == -1:
            break
        (flow, dist, dirt, holes) = move_dirt(x, y, 
                                              dirt, i_from, 
                                              holes, i_to)
        tot_work += flow * dist
    return tot_work

x = np.array([0, 1, 2, 3])          # p bins
y = np.array([2, 3, 4, 5])          # q bins
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4])

# Compute Wasserstein distance
wd = my_wasserstein(x, y, p.copy(), q.copy())
print("\nWasserstein distance = {:.2f}". format(wd))

# Compute Wasserstein distance using scipy.
wd = wasserstein_distance(x, y, p, q)
print("Wasserstein distance using scipy = {:.2f}".format(wd))
