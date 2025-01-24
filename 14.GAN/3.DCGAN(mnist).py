# [MXDL-14-03] 3.DCGAN(mnist).py
# A small version of DCGAN that outputs MNIST images of size 28 by 28 by 1.
#
# Source of D, G model: Advanced Deep Learning with Keras by Rowel Atienza (Chap. 4)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/k7cWq50CepY
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# Load MNIST data set
with open('data/mnist.pkl', 'rb') as f:
        x, y = pickle.load(f)  # x: (70000, 784), y: (70000, 1)

x_real = x[:10000].reshape(-1, 28, 28, 1)

# Build a discriminator model
D_in = Input(batch_shape = (None, 28, 28, 1))
dh = Conv2D(32, 5, strides=1, activation='leaky_relu', padding='same')(D_in)
dh = Conv2D(64, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Conv2D(128, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Conv2D(256, 5, strides=2, activation='leaky_relu', padding='same')(dh)
dh = Flatten()(dh)
D_out = Dense(1, activation='sigmoid')(dh)
D = Model(D_in, D_out)
D.summary()

# Build a generator model
nG_input = 100
G_in = Input(batch_shape=(None, nG_input))
gh = Dense(7 * 7 * 128)(G_in)
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
G_out = Activation('sigmoid')(gh)
G = Model(G_in, G_out)
G.summary()

# Build a DCGAN model using a Keras custom model
# reference: https://www.tensorflow.org/guide/keras
class DCGAN(Model):
    def __init__(self, D, G, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.G = G
        self.D_opt = optimizers.Adam(0.0002, beta_1 = 0.5)
        self.G_opt = optimizers.Adam(0.0002, beta_1 = 0.5)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")
    
    # Loss for Discriminator
    def lossD(self, x, z):
        # Clip the output values of D to prevent the loss from
        # becoming non or inf.
        Gz = self.G(z)
        Dx = tf.clip_by_value(self.D(x), 1e-8, 1.0)
        DGz = tf.clip_by_value(self.D(Gz), 1e-8, 0.999999)
        return -tf.reduce_mean(tf.math.log(Dx) + tf.math.log(1. - DGz))

    # Loss for Generator
    def lossG(self, z):
        Gz = self.G(z)
        DGz = tf.clip_by_value(self.D(Gz), 1e-8, 1.0)
        return -tf.reduce_mean(tf.math.log(DGz))
    
    @property
    def metrics(self):
        return [
            self.D_loss_tracker,
            self.G_loss_tracker,
        ]

    def train_step(self, x):
        m = tf.shape(x)[0]     # minibatch size
        
        # Sample minibatch of m noise samples z from
        # a uniform distribution
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        
        # Update the discriminator D
        self.D_opt.minimize(lambda: self.lossD(x, z), self.D.trainable_variables)
        
        # Sample minibatch of m noise samples z from
        # a uniform distribution
        z = tf.random.uniform((m, nG_input), -1.0, 1.0)
        
        # Update the generator G
        self.G_opt.minimize(lambda: self.lossG(z), self.G.trainable_variables)
        
        # Compute our own metrics
        self.D_loss_tracker.update_state(self.lossD(x, z))
        self.G_loss_tracker.update_state(self.lossG(z))
        
        return {
            "D_loss": self.D_loss_tracker.result(),
            "G_loss": self.G_loss_tracker.result(),
        }

# Create and compile a DCGAN model.
model = DCGAN(D, G)
model.compile(optimizer=Adam())
hist = model.fit(x_real, epochs=100, 
                 batch_size=200, shuffle=True, verbose=2)

# Loss history
plt.plot(hist.history['D_loss'], c='red'); plt.show()
plt.plot(hist.history['G_loss'], c='blue'); plt.show()

# Generate MNIST images
n_sample = 10
z = np.random.uniform(-1.0, 1.0, size=[n_sample, nG_input])
gen_img = G(z).numpy()

fig, ax = plt.subplots(1, n_sample, figsize=(14, 4))
for i in range(n_sample):
    p = gen_img[i, :, :]
    ax[i].imshow(p, cmap='gray')
    ax[i].axis('off')
plt.show()

