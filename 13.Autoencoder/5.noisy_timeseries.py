# [MXDL-13-02] 5.noisy_timeseries.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Q8AyijWiJyk
#
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Generate training data: 2 noisy sine curves
n = 10000       # the number of data points
n_step = 50     # the number of time steps

# Generate two sine curves
s1 = np.sin(np.pi * 0.06 * np.arange(n))
s2 = 0.5*np.sin(np.pi * 0.05 * np.arange(n))
s = np.vstack([s1, s2]).T  # shape = (1000, 2)

# Add noise to two sine curves
x1 = s1 + np.random.random(n)
x2 = s2 + np.random.random(n)
x = np.vstack([x1, x2]).T  # shape = (1000, 2)

# Generate training data for the LSTM denoising autoencoder
# x_data: noised sine curves
# y_data: clean sine curves
# x_data, y_data.shape = (970, 30, 2)
m = np.arange(0, n - n_step)
x_data = np.array([x[i:(i+n_step), :] for i in m])
y_data = np.array([s[i:(i+n_step), :] for i in m])

# Visualize
i = np.random.choice(x_data.shape[0], 1)[0]
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(x_data[i], linewidth=3)
ax[0].set_title("Noised sine curves")
ax[1].plot(y_data[i], linewidth=3)
ax[1].set_title("Denoised sine curves")
plt.show()

# Save the training data
with open('data/noisy_sine.pkl', 'wb') as f:
	pickle.dump([x_data, y_data], f)
