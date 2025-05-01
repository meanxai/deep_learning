# [MXDL-14-11] 12.D2GAN.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/tQz9_NR91pQ
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

nD_input = x_real.shape[1]
nD_hidden = 512
nD_output = 1
nG_input = 50
nG_hidden = 512
nG_output = nD_input

# Create a Discriminator model
def Discriminator():
    D_in = Input(batch_shape=(None, nD_input))
    D_Ho1 = Dense(nD_hidden, activation='leaky_relu')(D_in)
    D_Ho2 = Dense(nD_hidden, activation='leaky_relu')(D_Ho1)
    D_Out = Dense(nD_output, activation='softplus')(D_Ho2)
    return Model(D_in, D_Out)

# Create a Generator model
def Generator():
    G_in = Input(batch_shape=(None, nG_input))
    G_Ho1 = Dense(nG_hidden, activation='leaky_relu')(G_in)
    G_Ho2 = Dense(nG_hidden, activation='leaky_relu')(G_Ho1)
    G_Out = Dense(nG_output)(G_Ho2)
    return Model(G_in, G_Out)

D1 = Discriminator()
D2 = Discriminator()
G = Generator()

# Build a D2GAN model, a Keras custom model
class D2GAN(Model):
    def __init__(self, D1, D2, G, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.D1 = D1
        self.D2 = D2
        self.G = G
        self.alpha = alpha
        self.beta = beta
        self.D1_opt = Adam(learning_rate = 0.0005, beta_1=0.5, beta_2=0.9)
        self.D2_opt = Adam(learning_rate = 0.0005, beta_1=0.5, beta_2=0.9)
        self.G_opt = Adam(learning_rate = 0.0005, beta_1=0.5, beta_2=0.9)
        self.D1_loss_tracker = Mean(name="D1_loss")
        self.D2_loss_tracker = Mean(name="D2_loss")
        self.G_loss_tracker = Mean(name="G_loss")
    
    # Compute the loss of D2GAN
    def compute_J(self, x, z):
        Gz = self.G(z)
        D1x = self.D1(x)
        D1Gz = self.D1(Gz)
        D2x = self.D2(x)
        D2Gz = self.D2(Gz)
        J = self.alpha * tf.math.log(D1x + 1e-8) - D1Gz + \
            self.beta * tf.math.log(D2Gz + 1e-8) - D2x
        return J
        
    def lossD1(self, x, z):
        J = self.compute_J(x, z)
        return -tf.reduce_mean(J)
        
    def lossD2(self, x, z):
        J = self.compute_J(x, z)
        return -tf.reduce_mean(J)

    def lossG(self, x, z):
        J = self.compute_J(x, z)
        return tf.reduce_mean(J)
    
    @property
    def metrics(self):
        return [self.D1_loss_tracker,
                self.D2_loss_tracker,
                self.G_loss_tracker]

    def train_step(self, x):
        # Update the discriminators, D1 and D2 2 times.
        m = tf.shape(x)[0]  # batch size
        for i in range(2):
            z = tf.random.uniform((m, nG_input), -1.0, 1.0)
            self.D1_opt.minimize(lambda: self.lossD1(x, z), 
                                 self.D1.trainable_variables)
            self.D2_opt.minimize(lambda: self.lossD2(x, z), 
                                 self.D2.trainable_variables)
        
        # Update the generator
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        self.G_opt.minimize(lambda: self.lossG(x, z), 
                            self.G.trainable_variables)
        
        # Compute our own metrics
        self.D1_loss_tracker.update_state(self.lossD1(x, z))
        self.D2_loss_tracker.update_state(self.lossD2(x, z))
        self.G_loss_tracker.update_state(self.lossG(x, z))
        
        return {
            "D1_loss": self.D1_loss_tracker.result(),
            "D2_loss": self.D2_loss_tracker.result(),
            "G_loss": self.G_loss_tracker.result(),
        }

# Create and compile a D2GAN model.
model = D2GAN(D1, D2, G, alpha=1.0, beta=1.0)
model.compile(optimizer=Adam())

# Observe the distribution of the real data x and 
# the fake data G(z).
def plot_distribution():
    z = np.random.uniform(-1.0, 1.0, (x_real.shape[0], nG_input))
    x_fake = model.G(z).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_real[:, 0], x_real[:, 1], c='blue', alpha=0.5, 
                s=5, label='real')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], c='red', alpha=0.5, 
                s=5, label='fake')
    plt.legend()
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)    
    plt.title('Distribution of real and fake data')
    plt.show()

D1_loss = []
D2_loss = []
G_loss = []
for i in range(20):
    loss = model.fit(x_real, epochs=50, batch_size=200, verbose=0)
    D1_loss += loss.history['D1_loss']
    D2_loss += loss.history['D1_loss']
    G_loss += loss.history['G_loss']
    p = '\nepochs = {}, D1_loss = {:.2f}, D2_loss = {:.2f}, G_loss = {:.2f}'
    print(p.format(50 + i * 50, D1_loss[-1], D2_loss[-1], G_loss[-1]), end='')
    plot_distribution()

# Loss history
plt.plot(D1_loss, c='red')
plt.title('Discriminator-1 loss')
plt.show()

plt.plot(D1_loss, c='green')
plt.title('Discriminator-2 loss')
plt.show()

plt.plot(G_loss, c='blue')
plt.title('Generator loss')
plt.show()

D1.predict(x_real, verbose=0)
D2.predict(x_real, verbose=0)
