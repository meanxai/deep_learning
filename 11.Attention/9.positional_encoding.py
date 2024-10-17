# [MXDL-11-05] 9.positional_encoding.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/k0iKqtzz6KI
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# reference: https://github.com/suyash/transformer
def positional_encoding(position, d_model):
    position_dims = np.arange(position)[:, np.newaxis]
    embed_dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000.0, (2 * (embed_dims//2))/d_model)
    angle_rads = position_dims * angle_rates
    
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    return np.concatenate([sines, cosines], axis=-1)

PE = positional_encoding(6,8)
print(np.round(PE, 3), '\n')

for i in range(PE.shape[0] - 1):
    d = euclidean_distances(PE[i].reshape(1,-1), PE[i+1].reshape(1,-1))
    norm = np.linalg.norm(PE[i])
    dot = np.dot(PE[i], PE[i+1])
    print("%d - %d : distance = %.4f, norm = %.4f, dot = %.4f" % (i, i+1, d[0,0], norm, dot))

PE = positional_encoding(20, 2)
plt.figure(figsize=(4,4))
plt.plot(PE[:,0], PE[:,1], marker='o', linewidth=1.0, color='red')
plt.show()

PE = positional_encoding(20, 3)
fig = plt.figure(figsize=(3,3), dpi=100)  
ax = fig.add_subplot(1,1,1, projection='3d')  
ax.plot(PE[:,0], PE[:,1], PE[:, 2],
        marker='o', markersize=4, linewidth=1.0, color='red')
plt.show()
