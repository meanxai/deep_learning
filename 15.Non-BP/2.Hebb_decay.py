# [MXDL-15-02] 2.Hebb_decay.py
# A simple example of pattern recognition using Hebb's rule
# with weight decay
import numpy as np

# Training stage
# Input data: shape = (3, 9)
x = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 0],  # the letter 'T'
              [1, 0, 1, 1, 0, 1, 1, 1, 1],  # the letter 'U'
              [1, 1, 1, 1, 0, 0, 1, 1, 1]]) # the letter 'C'

# Output data: shape = (3, 3)
y = np.array([0, 1, 2])
labels = np.array(['T', 'U', 'C'])

# Initialize the connection weights
w = np.zeros((y.shape[0], x.shape[1]))  # (3, 9)

# Update w using Hebb's rule with weight decay
alpha = 0.1  # learning rate
for k in range(100):
    for xi, yi in zip(x, y):
        w[yi, :] += alpha * (xi - w[yi, :])

print("\nWeights (w):\n\n{}". format(w.T.round(2)))

# Test stage
# Corrupted images: shape = (3, 9)
xt = np.array([[ 1, 0, 1, 0, 1, 0, 0, 1, 0],    # corrupted 'T'
               [ 0, 1, 1, 1, 0, 0, 0, 1, 1],    # corrupted 'C'
               [ 1, 0, 1, 0, 0, 1, 1, 1, 0]])   # corrupted 'U'

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# For each corrupted image, we determine which neuron 
# in the output layer is activated.
yt = softmax(xt @ w.T)
print("\nyt = xt @ w: \n{}".format(yt.round(2)))

y_hat = np.argmax(yt, axis=1)

# Predictions:
print("\nPrediction results:")
for i in range(xt.shape[0]):
    print("\n{} --> It looks like the letter '{}'."\
          .format(xt[i].reshape(3,3), labels[y_hat[i]]))
