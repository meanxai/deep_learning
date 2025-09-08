# [MXDL-15-09] 11.RBM.py
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
    x_data, _ = pickle.load(f)     # x_data: (70000, 784)

x_data = x_data[:100]
x_data = np.where(x_data > 0.2, 1, 0)
n, m = x_data.shape   # n=100, m=784

NV = m     # the number of visible neurons
NH = 500   # the number of hidden neurons
RL = 0.01  # Learning rate
CD_K = 1   # one step Contrastive Divergence

def sampling(p):
    return np.random.binomial(n=1, p=p)

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Initialize the model parameters in an RBM
w = np.random.normal(0, 0.01, size=(NV, NH))  # weights (784, 500)
b = np.zeros([NV])                   # biases for visible neurons
c = np.zeros([NH])                   # biases for hidden neurons

n_iter = 2000     # the number of iterations
n_size = 20       # a mini-batch size
n_batch = int(x_data.shape[0] / n_size) # the number of mini-batches
loss = []
for epoch in range(n_iter):
    for k in range(n_batch):
        # Yield a mini-batch
        start = k * n_size
        end = start + n_size
        bx = x_data[start:end]
        
        # K-steps Contrastive Divergence
        v = bx.copy()        # the starting point
        for i in range(CD_K):
            h = sigmoid(np.dot(v, w) + c)     # p(h=1 | v)
            h = sampling(h)                   # h ∈ {0, 1}, (20, 500)
            
            v = sigmoid(np.dot(h, w.T) + b)   # p(v=1 | h)
            v = sampling(v)                   # v ∈ {0, 1}, (20, 784)
        
        # E_data
        h_data = sigmoid(np.dot(bx, w) + c)
        E_data = np.dot(bx.T, h_data) / n_size
        
        # E_model
        h_model = sigmoid(np.dot(v, w) + c)
        E_model = np.dot(v.T, h_model) / n_size

        # update parameters
        w += RL * (E_data - E_model)
        b += RL * np.mean(bx - v, axis=0)
        c += RL * np.mean(h_data - h_model, axis=0)
    
    if epoch % 10 == 0:
        loss.append(np.mean(np.square(bx - v)))
        print("epochs: {}, loss: {:.4f}".format(epoch, loss[-1]))

# Plot the loss change
plt.plot(loss, color='red')
plt.show()

# Reconstruct an image vector x
def reconstruct(x):
    h = sigmoid(np.dot(x, w) + c)
    h = sampling(h) 
    
    v = sigmoid(np.dot(h, w.T) + b)
    v = np.where(v > 0.2, 1, 0)
    
    fig = plt.figure(figsize=(5,2))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    x_org = x.reshape(28,28)
    ax1.imshow(x_org)
    ax1.set_title("corrupted image")
    ax1.axis('off')
    
    x_hat = v.reshape(28,28)
    ax2.imshow(x_hat)
    ax2.set_title("reconstructed image")
    ax2.axis('off')
    plt.show()

def corrupted_img(p):
    p.reshape(28,28)[18:, :] = 0
    return p.reshape(-1,)

# Reconstruct 10 image patterns
for i in range(10):
    x = x_data[i].reshape(1, -1)
    reconstruct(corrupted_img(x))
