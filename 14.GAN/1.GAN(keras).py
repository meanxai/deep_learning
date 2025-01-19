# [MXDL-14-02] 1.GAN(keras).py
# Implementation of a GAN model for generating 2D data with 3 clusters
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Hu4GtiJRmbI
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a two-dimensional real data set x, consisting of three clusters
x_real, _ = make_blobs(n_samples = 1500, n_features = 2, 
                        centers=[(0., 0), (0.5, 0.5), (1., 0.)], 
                        cluster_std = 0.15)
    
nD_input = x_real.shape[1]
nD_hidden = 64
nD_output = 1
nG_input = 50
nG_hidden = 64
nG_output = nD_input

# Build a Discriminator model
D_in = Input(batch_shape=(None, nD_input))
D_Ho1 = Dense(nD_hidden, activation='relu')(D_in)
D_Ho2 = Dense(nD_hidden, activation='relu')(D_Ho1)
D_Out = Dense(nD_output, activation='sigmoid')(D_Ho2)
D = Model(D_in, D_Out)

# Build a Generator model
G_in = Input(batch_shape=(None, nG_input))
G_Ho1 = Dense(nG_hidden, activation='relu')(G_in)
G_Ho2 = Dense(nG_hidden, activation='relu')(G_Ho1)
G_Out = Dense(nG_output)(G_Ho2)
G = Model(G_in, G_Out)

# Build a GAN model using a Keras custom model
class GAN(Model):
    def __init__(self, D, G, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.D = D
        self.G = G
        self.D_opt = Adam(0.0002, beta_1 = 0.5)
        self.G_opt = Adam(0.0002, beta_1 = 0.5)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")
    
    # The loss for Discriminator
    def lossD(self, x, z):
        Gz = self.G(z)
        Dx = self.D(x)
        DGz = self.D(Gz)
        return -tf.reduce_mean(tf.math.log(Dx + 1e-8) + tf.math.log(1 - DGz + 1e-8))

    # The loss for Generator
    def lossG(self, z):
        Gz = self.G(z)
        DGz = self.D(Gz)
        return tf.reduce_mean(tf.math.log(1 - DGz + 1e-8))
    
    @property
    # We list our 'Metric' objects here so that 'reset_states()' 
    # can be called automatically at the start of each epoch
    # or at the start of 'evaluate()'.  
    def metrics(self):
        return [
            self.D_loss_tracker,
            self.G_loss_tracker,
        ]

    # When you need to customize what fit() does, you should 
    # override the training step function of the Model class. This 
    # is the function that is called by fit() for every batch of 
    # data. You will then be able to call fit() as usual -- and it 
    # will be running your own learning algorithm.
    # In the body of the train_step method, we implement a regular
    # training update.
    def train_step(self, x):
        m = tf.shape(x)[0]       # minibatch size
        for k in range(self.k):
            # Sample minibatch of m noise samples z from a uniform distribution
            z = tf.random.uniform((m, nG_input), -1.0, 1.0)
            
            # Update the discriminator D
            self.D_opt.minimize(lambda: self.lossD(x, z), self.D.trainable_variables)
        
        # Sample minibatch of m noise samples z from a uniform distribution
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        
        # Update the generator G
        self.G_opt.minimize(lambda: self.lossG(z), self.G.trainable_variables)
        
        # Compute our own metrics
        self.D_loss_tracker.update_state(self.lossD(x, z))
        self.G_loss_tracker.update_state(self.lossG(z))
        
        return {"D_loss": self.D_loss_tracker.result(),
                "G_loss": self.G_loss_tracker.result()}

# Create and compile a GAN model.
model = GAN(D, G, k=1)
model.compile(optimizer=Adam())

# Observe the distribution of the real data x and 
# the fake data G(z).
def plot_distribution():
    z = np.random.uniform(-1.0, 1.0, (x_real.shape[0], nG_input))
    x_fake = model.G(z).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_real[:, 0], x_real[:, 1], c='blue', alpha=0.5, s=5, label='real')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], c='red', alpha=0.5, s=5, label='fake')
    plt.legend()
    plt.title('Distibution of real and fake data')
    plt.show()
    
# Fit the GAN model to the real data
hist = model.fit(x_real, epochs=300, batch_size=100, verbose=2)

# Loss history
plt.plot(hist.history['D_loss'], c='red'); plt.show()
plt.plot(hist.history['G_loss'], c='blue'); plt.show()

# Plot the distribution of x and G(z)
plot_distribution()

# Verify that both D*(x) and D*(G(z)) converge to 0.5.
z = np.random.uniform(-1.0, 1.0, (x_real.shape[0], nG_input))
x_fake = model.G(z).numpy()
Dx = model.D(x_real).numpy()
DGz = model.D(x_fake).numpy()
print("D*(x):"); print(Dx)
print()
print("D*(G(z)):");print(DGz)

# To observe how the G(z) changes as training progresses, 
# run the code below.
# for i in range(5):
#     model.fit(x_real, epochs=60, batch_size=100, verbose=0)
#     print('\nepochs =', 60 + i * 60, end='')
#     plot_distribution()
