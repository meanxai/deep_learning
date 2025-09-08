# [MXDL-15-10] 12.ContrastiveHebb.py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split
import pickle

# Load the MNIST dataset
# x_data: (70000, 784), x_data: (70000, 1)
with open('data/mnist.pkl', 'rb') as f:
    x_input, y_target = pickle.load(f)

y_target = np.eye(10)[y_target].reshape(-1, 10) # one hot encoding
x_train, x_test, y_train, y_test = train_test_split(x_input, y_target)

ETA = 0.01                  # learning rate
GAMMA = 0.5                 # feedback gain
n_neurons = [784, 128, 10]  # the number of neurons in each layer
L = len(n_neurons) - 1      # L = 2

n_epochs = 10               # the number of iterations
b_size = 10                 # a mini-batch size
n_batches = int(x_train.shape[0] / b_size) # the number of mini-batches
T = 10                      # simulation time
dt = 0.5                    # Forward Euler method's time step

# Initialize the parameters: w[0], w[1], b[0], b[1]
w = [np.random.normal(0, 0.01, size=(i, o))
     for i, o in zip(n_neurons[:-1], n_neurons[1:])]
b = [np.random.normal(0, 0.01, size=(1, i))
     for i in n_neurons[1:]]

def forward_phase(x_input):
    x = [np.zeros((x_input.shape[0], m)) for m in n_neurons]
    x[0] = x_input   # input data
    
    for t in np.arange(1, T+1, dt):
        for k in range(1, L+1):   # k = 1, 2
            activation = np.dot(x[k-1], w[k-1])
            if k < L:
                activation += GAMMA * np.dot(x[k+1], w[k].T)
            activation += b[k-1]
            x[k] += dt * (-x[k] + sigmoid(activation))
    return x

def backward_phase(x_input, y_target):
    x = [np.zeros((x_input.shape[0], m)) for m in n_neurons]
    x[0] = x_input     # input data
    x[-1] = y_target   # target data
    
    for t in np.arange(1, T+1, dt):
        for k in range(1, L):  # k = 1
            activation = np.dot(x[k-1], w[k-1]) +\
                 GAMMA * np.dot(x[k+1], w[k].T) + b[k-1]
            x[k] += dt * (-x[k] + sigmoid(activation))
    return x

for e in range(n_epochs):
    for j in tqdm(range(n_batches), desc=f"Epoch {e+1}/{n_epochs}"):
        b_start = j * b_size
        b_end = b_start + b_size
        
        x_batch = x_train[b_start:b_end]
        y_batch = y_train[b_start:b_end]
        
        # Forward and backward phases
        x_fwd = forward_phase(x_batch)
        x_bwd = backward_phase(x_batch, y_batch)
        
        # Update parameters
        for k in range(L):    # k = 0, 1
            C = ETA * GAMMA ** (k - L)
            w[k] += C * (np.dot(x_bwd[k].T, x_bwd[k+1]) -
                         np.dot(x_fwd[k].T, x_fwd[k+1]))
            b[k] += C * np.mean(x_bwd[k+1] - x_fwd[k+1], axis=0)

# get predictions on test set
x_fwd = forward_phase(x_test)      
y_pred = np.argmax(x_fwd[-1], axis=1)
y_true = np.argmax(y_test, axis=1)

acc = (y_true == y_pred).mean()
print('\nAccuracy on test data = {:.4f}'.format(acc))

