# [MXDL-15-01] 1.Hebb_rule.py
# A simple example of the Hebbian learning rule for pattern 
# recognition
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/M6wRIqug93Y
#
import numpy as np

# Input data: shape = (3, 9)
x = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 0],  # the letter 'T'
              [1, 0, 1, 1, 0, 1, 1, 1, 1],  # the letter 'U'
              [1, 1, 1, 1, 0, 0, 1, 1, 1]]) # the letter 'C'

# Output data: shape = (3, 3)
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
labels = np.array(['T', 'U', 'C'])

# Initialize the connection weights
w = np.zeros((x.shape[1],y.shape[0]))  # (9, 3)

# Update w with the Hebbian rule
alpha = 0.1  # learning rate
alpha = 1
for i in range(5):
    for i in range(x.shape[0]):
        w += alpha * np.outer(x[i], y[i])
        w = (w - w.mean()) / np.std(w)    # Normalize
    print("\nWeights (w):\n\n{}".format(w.round(2)))

# Test data: corrupted images, shape = (3, 9)
xt = np.array([[ 1, 0, 1, 0, 1, 0, 0, 1, 0], # corrupted 'T'
               [ 0, 1, 1, 1, 0, 0, 0, 1, 1], # corrupted 'C'
               [ 1, 0, 1, 0, 0, 1, 1, 1, 0]])# corrupted 'U'

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# For each corrupted image, we determine which neuron 
# in the output layer is activated.
yt = softmax(xt @ w)
print("\ny = x @ w: \n{}".format(yt.round(2)))

y_hat = np.argmax(yt, axis=1)

# Predictions:
print("\nPrediction results:")
for i in range(xt.shape[0]):
    print("\n{} --> It looks like the letter '{}'."\
          .format(xt[i].reshape(3,3), labels[y_hat[i]]))
