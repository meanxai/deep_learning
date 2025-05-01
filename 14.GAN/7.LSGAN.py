# [MXDL-14-07] 7.LSGAN.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/XWCMY_Cx9A0
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a dataset consisting of eight Gaussian distributions 
# arranged in a circle.
cent = [(-0.4, 0), (-0.285, -0.285), (-0.285, 0.285), (0., -0.4), 
        (0., 0.4), (0.285, -0.285), (0.285, 0.285), (0.4, 0)]
x_real, _ = make_blobs(n_samples = 1600, n_features = 8, 
                        centers = cent, 
                        cluster_std = 0.04)

nD_input = x_real.shape[1]
nD_hidden = 128
nD_output = 1
nG_input = 50
nG_hidden = 128
nG_output = nD_input

# Build a Discriminator model
d_in = Input(batch_shape=(None, nD_input))
dh = Dense(nD_hidden, activation='relu')(d_in)
dh = Dense(nD_hidden, activation='relu')(dh)
d_out = Dense(nD_output)(dh)  # linear activation
D = Model(d_in, d_out)

# Build a Generator model
g_in = Input(batch_shape=(None, nG_input))
gh = Dense(nG_hidden, activation='relu')(g_in)
gh = Dense(nG_hidden, activation='relu')(gh)
g_out = Dense(nG_output)(gh)
G = Model(g_in, g_out)

# Build an LSGAN model using a Keras custom model
class LSGAN(Model):
    def __init__(self, D, G, a=0., b=1., c=1., **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.G = G
        self.a = a
        self.b = b
        self.c = c
        self.D_opt = RMSprop(0.002, rho=0.99)
        self.G_opt = RMSprop(0.002, rho=0.99)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")
    
    # Discriminator loss
    def lossD(self, x, z):
        Gz = self.G(z)
        Dx = self.D(x)
        DGz = self.D(Gz)
        return 0.5 * tf.reduce_mean(tf.square(Dx - self.b) + 
                                    tf.square(DGz - self.a))

    # Generator loss
    def lossG(self, z):
        Gz = self.G(z)
        DGz = self.D(Gz)
        return 0.5 * tf.reduce_mean(tf.square(DGz - self.c))
           
    @property
    def metrics(self):
        return [self.D_loss_tracker, self.G_loss_tracker]

    def train_step(self, x):
        n_batch = tf.shape(x)[0]
        
        # Update the discriminator
        for i in range(2):
            z = tf.random.uniform((n_batch, nG_input), -1.0, 1.0)
            self.D_opt.minimize(lambda: self.lossD(x, z), self.D.trainable_variables)
        
        # Update the generator
        z = tf.random.uniform((n_batch, nG_input), -1.0, 1.0)
        self.G_opt.minimize(lambda: self.lossG(z), self.G.trainable_variables)
        
        # Compute our own metrics
        self.D_loss_tracker.update_state(self.lossD(x, z))
        self.G_loss_tracker.update_state(self.lossG(z))
        
        return {"D_loss": self.D_loss_tracker.result(),
                "G_loss": self.G_loss_tracker.result()}

# Create and compile an LSGAN model.
model = LSGAN(D, G, a=0, b=1, c=1)
model.compile(optimizer=RMSprop())

# Observe the distribution of the real data x and 
# the fake data G(z).
def plot_distribution():
    z = tf.random.uniform((tf.shape(x_real)[0], nG_input), -1.0, 1.0)
    x_fake = model.G(z).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_real[:, 0], x_real[:, 1], c='blue', alpha=0.5, s=5, label='real')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], c='red', alpha=0.5, s=5, label='fake')
    plt.legend()
    plt.title('Distibution of real and fake data')
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
