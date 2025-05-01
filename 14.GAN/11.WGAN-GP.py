# [MXDL-14-10] 11.WGAN-GP.py
# WGAN with Gradient Penalty
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/kGANAq7_oKI
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a dataset consisting of eight Gaussian distributions 
# arranged in a circle.
cent = [(-0.4, 0), (-0.285, -0.285), (-0.285, 0.285), (0., -0.4), 
        (0., 0.4), (0.285, -0.285), (0.285, 0.285), (0.4, 0)]
x_real, _ = make_blobs(n_samples = 1600, n_features = 8, 
                        centers = cent, 
                        cluster_std = 0.04)

# Network parameters
nD_input = x_real.shape[1] # the number of inputs to the discriminator
nD_hidden = 512      # the number of hidden neurons in the discriminator
nD_output = 1        # the number of outputs from the discriminator
nG_input = 50        # the number of inputs to the generator
nG_hidden = 512      # the number of hidden neurons in the generator
nG_output = nD_input # the number of outputs from the generator

# Build a Discriminator (Critric) model
d_in = Input(batch_shape=(None, nD_input))
dh = Dense(nD_hidden, activation='relu')(d_in)
dh = Dense(nD_hidden, activation='relu')(dh)
dh = Dense(nD_hidden, activation='relu')(dh)
d_out = Dense(nD_output)(dh)
D = Model(d_in, d_out)

# Build a Generator model
g_in = Input(batch_shape=(None, nG_input))
gh = Dense(nG_hidden, activation='relu')(g_in)
gh = Dense(nG_hidden, activation='relu')(gh)
gh = Dense(nG_hidden, activation='relu')(gh)
g_out = Dense(nG_output)(gh)
G = Model(g_in, g_out)

# Build a WGAN-GP model using a Keras custom model
class WGAN_GP(Model):
    def __init__(self, D, G, gp_lambda=10.0, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.G = G
        self.gp_lambda = gp_lambda   # gradient penalty coefficient
        self.D_opt=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.G_opt=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")

    def lossD(self, x, z):
        Gz = self.G(z)     # the output of the generator for z
        Dx = self.D(x)     # the output of the discriminator for x
        DGz = self.D(Gz)   # the output of the discriminator for Gz
        m = tf.shape(x)[0] # the batch size of training data
        
        # Compute the loss with gradient penalty
        e = tf.random.uniform((m, 1), 0., 1.)
        x_hat = e * x + (1. - e) * Gz                        
        Dx_hat = self.D(x_hat)
        grads = tf.gradients(Dx_hat, [x_hat])[0]
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = self.gp_lambda * tf.reduce_mean((norm2 - 1.0) ** 2)
        return -tf.reduce_mean(Dx) + tf.reduce_mean(DGz) + gp
    
    def lossG(self, z):
        Gz = self.G(z)
        DGz = self.D(Gz)
        return -tf.reduce_mean(DGz)
    
    @property
    def metrics(self):
        return [self.D_loss_tracker, self.G_loss_tracker]

    def train_step(self, x):
        # Update the discriminator 5 times.
        m = tf.shape(x)[0]  # batch size
        for i in range(5):
            z = tf.random.uniform((m, nG_input), -1.0, 1.0)
            self.D_opt.minimize(lambda: self.lossD(x, z), self.D.trainable_variables)

        # Update the generator
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        self.G_opt.minimize(lambda: self.lossG(z), self.G.trainable_variables)
        
        # Compute our own metrics
        self.D_loss_tracker.update_state(self.lossD(x, z))
        self.G_loss_tracker.update_state(self.lossG(z))
        
        return {
            "D_loss": self.D_loss_tracker.result(),
            "G_loss": self.G_loss_tracker.result(),
        }

# Create and compile a WGAN model.
model = WGAN_GP(D, G, gp_lambda = 0.1)
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
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)    
    plt.title('Distribution of real and fake data')
    plt.show()

D_loss = []
G_loss = []
for i in range(20):
    loss = model.fit(x_real, epochs=50, batch_size=200, verbose=0)
    D_loss += loss.history['D_loss']
    G_loss += loss.history['G_loss']
    p = '\nepochs = {}, D_loss = {:.2f}, G_loss = {:.2f}'
    print(p.format(50 + i * 50, D_loss[-1], G_loss[-1]), end='')
    plot_distribution()

# Loss history
plt.plot(D_loss, c='red')
plt.title('Discriminator loss')
plt.show()

plt.plot(G_loss, c='blue')
plt.title('Generator loss')
plt.show()
