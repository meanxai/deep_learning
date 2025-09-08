# [MXDL-15-02] 3.Hebb(mnist).py
# MNIST image classification using Hebb's rule with weight decay
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Load MNIST data set
with open('data/mnist.pkl', 'rb') as f:
    x, y = pickle.load(f)     # x: (70000, 784), y: (70000, 1)

y = y.reshape(-1,)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Initialize the connection weights # (10, 784)
w = np.zeros((10, x_train.shape[1]))

# Update w using Hebb's rule with weight decay
alpha = 0.01     # learning rate
n_iters = 50     # number of iterations
for k in range(n_iters):
    for xi, yi in zip(x_train, y_train):
        w[yi, :] += alpha * (xi - w[yi, :])

    print("Epoch:", k+1)
    
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# For each corrupted image, we determine which neuron 
# in the output layer is activated.
y_prob = softmax(x_test @ w.T)
y_pred = np.argmax(y_prob, axis=1).reshape(-1,)
acc = (y_test == y_pred).mean()
print('Accuracy on the test data = {:.2f}'.format(acc))

# Let's check out some misclassified images.
# n_sample = 10
# miss_cls = np.where(y_test != y_pred)[0]
# miss_sam = np.random.choice(miss_cls, n_sample)

# fig, ax = plt.subplots(1, n_sample, figsize=(14,4))
# for i, miss in enumerate(miss_sam):
#     x = x_test[miss]
#     ax[i].imshow(x.reshape(28, 28))
#     ax[i].axis('off')
#     ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))

