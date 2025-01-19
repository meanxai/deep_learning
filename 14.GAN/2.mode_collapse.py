# [MXDL-14-02] 2.model_collapse.py
# Observe the mode collapse phenomenon in a standard GAN
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
from tensorflow.keras import optimizers
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
nD_input = x_real.shape[1]
nD_hidden = 128
nD_output = 1
nG_input = 50
nG_hidden = 128
nG_output = nD_input

# Build a Discriminator model
D_in = Input(batch_shape=(None, nD_input))
D_Ho = Dense(nD_hidden, activation='relu')(D_in)
D_Out = Dense(nD_output, activation='sigmoid')(D_Ho)
D = Model(D_in, D_Out)

# Build a Generator model
G_in = Input(batch_shape=(None, nG_input))
G_Ho = Dense(nG_hidden, activation='relu')(G_in)
G_Out = Dense(nG_output)(G_Ho)
G = Model(G_in, G_Out)

# Build a GAN model using a Keras custom model
class GAN(Model):
    def __init__(self, D, G, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.G = G
        self.D_opt = optimizers.Adam(0.0002, beta_1 = 0.5)
        self.G_opt = optimizers.Adam(0.0005, beta_1 = 0.5)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")
       
    def lossD(self, x, z):
        Gz = self.G(z)
        Dx = self.D(x)
        DGz = self.D(Gz)
        return -tf.reduce_mean(tf.math.log(Dx + 1e-8) + tf.math.log(1 - DGz + 1e-8))

    def lossG(self, z):
        Gz = self.G(z)
        DGz = self.D(Gz)
        return tf.reduce_mean(tf.math.log(1 - DGz + 1e-8))
    
    @property
    def metrics(self):
        return [
            self.D_loss_tracker,
            self.G_loss_tracker,
        ]

    def train_step(self, x):
        m = tf.shape(x)[0]
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        self.D_opt.minimize(lambda: self.lossD(x, z), self.D.trainable_variables)
        
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        self.G_opt.minimize(lambda: self.lossG(z), self.G.trainable_variables)
        
        self.D_loss_tracker.update_state(self.lossD(x, z))
        self.G_loss_tracker.update_state(self.lossG(z))
        
        return {
            "D_loss": self.D_loss_tracker.result(),
            "G_loss": self.G_loss_tracker.result(),
        }

# Create and compile a GAN model.
model = GAN(D, G)
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

hist = model.fit(x_real, epochs=1000, batch_size=100, verbose=2)

# Observe the mode collapse phenomenon.
for i in range(10):
    print('epochs =', 1000 + i * 50)
    model.fit(x_real, epochs=50, batch_size=100, verbose=0)
    plot_distribution()

