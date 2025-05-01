# [MXDL-14-07] 8.LS_DCGAN.py
# A small version of LS_DCGAN that outputs MNIST images 
# of size 28 by 28 by 1.
# Source of D, G model: 
# Advanced Deep Learning with Keras by Rowel Atienza (Chap. 4)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/XWCMY_Cx9A0
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# Load MNIST data set
with open('data/mnist.pkl', 'rb') as f:
        x, y = pickle.load(f)  # x: (70000, 784), y: (70000, 1)

x_real = x[:20000].reshape(-1, 28, 28, 1)
nG_input = 100

# Build a discriminator model
d_in = Input(batch_shape = (None, 28, 28, 1))
dh = Conv2D(32, 5, strides=1, activation='leaky_relu', padding='same')(d_in)
dh = Conv2D(64, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Conv2D(128, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Conv2D(256, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Flatten()(dh)
d_out = Dense(1, activation='sigmoid')(dh)
D = Model(d_in, d_out)

# Build a generator model
g_in = Input(batch_shape=(None, nG_input))
gh = Dense(7 * 7 * 128)(g_in)
gh = Reshape((7, 7, 128))(gh)
gh = Conv2DTranspose(128, 5, strides=2,padding='same')(gh)
gh = BatchNormalization()(gh)
gh = Activation('relu')(gh)
gh = Conv2DTranspose(64, 5, strides=2,padding='same')(gh)
gh = BatchNormalization()(gh)
gh = Activation('relu')(gh)
gh = Conv2DTranspose(32, 5, strides=1,padding='same')(gh)
gh = BatchNormalization()(gh)
gh = Activation('relu')(gh)
gh = Conv2DTranspose(1, 5, strides=1,padding='same')(gh)
gh = BatchNormalization()(gh)
g_out = Activation('sigmoid')(gh)
G = Model(g_in, g_out)

# Build an LS_DCGAN model using a Keras custom model
class LS_DCGAN(Model):
    def __init__(self, D, G, a=0., b=1., c=1., **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.G = G
        self.a = a
        self.b = b
        self.c = c
        self.D_opt = Adam(0.0005, beta_1 = 0.5)
        self.G_opt = Adam(0.0005, beta_1 = 0.5)
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

# Create and compile an LS_DCGAN model.
model = LS_DCGAN(D, G, a=0, b=1, c=1)
model.compile(optimizer=Adam())

# Fit the model to the real data
loss = model.fit(x_real, epochs=500, batch_size=500, verbose=2)

# Loss history
plt.plot(loss.history['D_loss'], c='red'); plt.show()
plt.plot(loss.history['G_loss'], c='blue'); plt.show()

# Generate MNIST images
n_sample = 10
z = tf.random.uniform((n_sample, nG_input), -1.0, 1.0)
gen_img = G(z).numpy()

fig, ax = plt.subplots(1, n_sample, figsize=(14, 4))
for i in range(n_sample):
    p = gen_img[i, :, :]
    ax[i].imshow(p, cmap='gray')
    ax[i].axis('off')
plt.show()


# D_loss = []
# G_loss = []
# D_loss += loss.history['D_loss']
# G_loss += loss.history['G_loss']
# plt.plot(D_loss, c='red')
# plt.title('Discriminator loss')
# plt.show()

# plt.plot(G_loss, c='blue')
# plt.title('Generator loss')
# plt.show()