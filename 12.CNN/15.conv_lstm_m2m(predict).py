# [MXDL-12-06] 15.conv_lstm_m2m(predict).py
# data : http://www.cs.toronto.edu/~nitish/unsupervised_video/
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/FzUAPtDgA_o
#
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load a moving MNIST dataset
with open('data/mv_mnist.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test, _ = pickle.load(f)

model = load_model("data/conv_lstm_m2m_1.h5")

# Predict a sequence of 5 images.
idx = np.random.choice(x_test.shape[0], 1)[0]
x_sample = np.expand_dims(x_test[idx], axis = 0) # (1, 19, 64, 64, 1)

p_result = []
for i in range(5):
    y_pred = model.predict(x_sample, verbose=0)[:,-1,:,:,:]  # (1, 64, 64, 1)
    p_result.append(y_pred)
    
    y_pred = np.expand_dims(y_pred, axis = 1)    # (1, 1, 64, 64, 1)
    x_sample = np.append(x_sample, y_pred, axis=1)[:,-19:,:,:]

# Plot the ground-truth image and a sequence of 5 predicted images.
fig, axes = plt.subplots(1, 6, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(np.squeeze(y_test[idx][-1]), cmap="gray")
        ax.set_title(f"Frame 20\nGround-truth")
    else:
        ax.imshow(np.squeeze(p_result[i-1]), cmap="gray")
        ax.set_title(f"Frame {20 + i - 1}\nPredicted")
    ax.axis("off")
plt.show()

		
