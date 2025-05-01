# [MXDL-14-05] 5.unrolledDCGAN.py
# A small version of unrolled DCGAN that outputs MNIST images 
# of size 28 by 28 by 1.
# Source of D, G model: 
# Advanced Deep Learning with Keras by Rowel Atienza (Chap. 4)
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/sOk90DUk5B4
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

x_real = x[:5000].reshape(-1, 28, 28, 1)
nG_input = 100

# Build a discriminator model
def Discriminator():
    D_in = Input(batch_shape = (None, 28, 28, 1))
    dh = Conv2D(32, 5, strides=1, activation='leaky_relu', padding='same')(D_in)
    dh = Conv2D(64, 5, strides=2, activation='leaky_relu', padding='same')(dh)
    dh = Conv2D(128, 5, strides=2, activation='leaky_relu', padding='same')(dh)
    dh = Conv2D(256, 5, strides=2, activation='leaky_relu', padding='same')(dh)
    dh = Flatten()(dh)
    D_out = Dense(1, activation='sigmoid')(dh)
    return Model(D_in, D_out)

# Build a generator model
def Generator():
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
    return Model(G_in, G_out)

D = Discriminator()  # discriminator (Ψ)
S = Discriminator()  # surrogate discriminator (θ)
G = Generator()      # generator (φ)

# Build an unrolled DCGAN model using a Keras custom model
class unrolledDCGAN(Model):
    def __init__(self, D, S, G, k, **kwargs):
        super().__init__(**kwargs)
        self.D = D   # discriminator model
        self.S = S   # surrogate discriminator model
        self.G = G   # generator model
        self.k = k   # the number of unrolling steps
        self.D_opt = optimizers.Adam(0.0002, beta_1 = 0.5)
        self.S_opt = optimizers.Adam(0.0002, beta_1 = 0.5)
        self.G_opt = optimizers.Adam(0.0005, beta_1 = 0.5)
        self.D_loss_tracker = Mean(name="D_loss")
        self.G_loss_tracker = Mean(name="G_loss")
    
    # Copy discriminator (Ψ) to surrogate discriminator (θ)
    def copy_variables(self, src, dest):
        for i in range(len(src.variables)):
            dest.variables[i].assign(src.variables[i])            

    # Discriminator loss
    # DS: real discriminator or surrogate discriminator
    def lossD(self, DS, x, z):
        # Clip the output values of D to prevent the loss from
        # becoming non or inf.
        Gz = self.G(z)
        Dx = tf.clip_by_value(DS(x), 1e-8, 1.0)
        DGz = tf.clip_by_value(DS(Gz), 1e-8, 0.999999)
        return -tf.reduce_mean(tf.math.log(Dx) + tf.math.log(1. - DGz))

    # Generator loss
    def lossG(self, DS, z):
        Gz = self.G(z)
        DGz = tf.clip_by_value(DS(Gz), 1e-8, 1.0)
        return -tf.reduce_mean(tf.math.log(DGz))
           
    @property
    def metrics(self):
        return [
            self.D_loss_tracker,
            self.G_loss_tracker,
        ]

    def train_step(self, x):
        # Update the discriminator (Ψ)
        z = tf.random.uniform((tf.shape(x)[0], nG_input), -1.0, 1.0)
        self.D_opt.minimize(lambda: self.lossD(self.D, x, z), self.D.trainable_variables)
        
        # Copy discriminator (Ψ) to surrogarate discriminator (θ)
        self.copy_variables(self.D, self.S)
        
        # Update the generator (φ)
        unrolled_grads = []
        z = tf.random.uniform((tf.shape(x)[0], nG_input), -1.0, 1.0)
        for i in range(self.k):
            # Update the surrogate discriminator (θ)
            self.S_opt.minimize(lambda: self.lossD(self.S, x, z), self.S.trainable_variables)
            
            with tf.GradientTape() as tape:
                unrolled_loss = self.lossG(self.S, z)
                
            grads = tape.gradient(unrolled_loss, self.G.trainable_variables)
            unrolled_grads.append(grads)
        
        mean_grads = []
        for g in zip(*unrolled_grads):
            mean_grads.append(tf.reduce_mean(g, axis=0))
            
        self.G_opt.apply_gradients(zip(mean_grads, self.G.trainable_variables))            
               
        # Compute our own metrics
        self.D_loss_tracker.update_state(self.lossD(self.D, x, z))
        self.G_loss_tracker.update_state(self.lossG(self.D, z))
        
        return {
            "D_loss": self.D_loss_tracker.result(),
            "G_loss": self.G_loss_tracker.result(),
        }

# Create and compile a unrolled DCGAN model.
model = unrolledDCGAN(D, S, G, k=5) # unrolling steps (k) = 5
model.compile(optimizer=Adam())

# Fit the model to the real data
hist = model.fit(x_real, epochs=500, batch_size=300, verbose=2)

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

