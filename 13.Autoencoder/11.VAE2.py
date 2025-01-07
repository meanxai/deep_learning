# [MXDL-13-06] 10.VAE2.py
# Implementation of a variational autoencoder
# source: https://keras.io/examples/generative/vae/
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Y5UAZ46rcuo
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle

class Sampling(Layer):
    # Uses mu (mean) and v (log var) to sample z
    def call(self, inputs):
        mu, v = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal([batch, dim])
        return mu + tf.exp(0.5 * v) * epsilon
    
# Build an encoder model
latent_dim = 2
e_input = Input(shape=(28, 28, 1))
x_enc = Conv2D(32, 3, activation="relu", strides=2, padding="same")(e_input)
x_enc = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x_enc)
x_enc = Flatten()(x_enc)
z_mean = Dense(latent_dim)(x_enc)     # mu
z_log_var = Dense(latent_dim)(x_enc)  # log variance (v)
z_z = Sampling()([z_mean, z_log_var])
encoder = Model(e_input, [z_mean, z_log_var, z_z])
encoder.summary()

# Build a decoder model
d_input = Input(shape=(latent_dim,))
x_dec = Dense(7 * 7 * 64, activation="relu")(d_input)
x_dec = Reshape((7, 7, 64))(x_dec)
x_dec = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x_dec)
x_dec = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x_dec)
y_dec = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x_dec) # (None, 28, 28, 1)
decoder = Model(d_input, y_dec)
decoder.summary()

# Build a custom VAE model
# reference: https://www.tensorflow.org/guide/keras
#                   /customizing_what_happens_in_fit
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.tot_loss_tracker = Mean(name="tot_loss")
        self.bce_loss_tracker = Mean(name="bce_loss")
        self.kld_loss_tracker = Mean(name="kld_loss")

    @property
    # We list our 'Metric' objects here so that 'reset_states()' 
    # can be called automatically at the start of each epoch
    # or at the start of 'evaluate()'.    
    def metrics(self):
        return [
            self.tot_loss_tracker, # total loss
            self.bce_loss_tracker, # binary cross entropy
            self.kld_loss_tracker, # KL divergence
        ]
    
    # When you need to customize what fit() does, you should 
    # override the training step function of the Model class. This 
    # is the function that is called by fit() for every batch of 
    # data. You will then be able to call fit() as usual -- and it 
    # will be running your own learning algorithm.
    # In the body of the train_step method, we implement a regular
    # training update.
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            d_output = self.decoder(z)
            
            # Computing loss
            # 1. binary crossentropy
            bce_loss = tf.reduce_mean(
                tf.reduce_sum(binary_crossentropy(data, d_output), axis=(1, 2))
            )
            
            # 2. KL divergence
            kld_loss = 0.5 * (tf.exp(log_var) + tf.square(mean) - log_var - 1)
            kld_loss = tf.reduce_mean(tf.reduce_sum(kld_loss, axis=1))
            tot_loss = bce_loss + kld_loss
        
        # Compute gradients and update weights
        grads = tape.gradient(tot_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.tot_loss_tracker.update_state(tot_loss)
        self.bce_loss_tracker.update_state(bce_loss)
        self.kld_loss_tracker.update_state(kld_loss)
        
        # Return a dict mapping metric names to current value
        return {
            "tot_loss": self.tot_loss_tracker.result(),
            "bce_loss": self.bce_loss_tracker.result(),
            "kld_loss": self.kld_loss_tracker.result(),
        }

# Load the MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)
x = x.reshape(-1, 28, 28, 1)
y = y.reshape(-1,)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train the VAE model
model = VAE(encoder, decoder)
model.compile(optimizer=Adam())
hist = model.fit(x_train, epochs=30, batch_size=500, verbose=2)

# Visually check the latent vectors in the z-space
z_list = []
for i in np.unique(y_test):
    x = x_test[y_test == i]
    
    z_mean, z_log_var, z = model.encoder.predict(x)
    z_list.append(z) 

plt.figure(figsize=(6,6))
for i in range(10):
    z = np.array(z_list[i])
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, label=str(i))
plt.legend()    
plt.show()

def plot_latent_space(vae, n=30, figsize=10):
    # display a n*n 2D manifold of digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-3, 1, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
        print('y-axis = {} done'.format(yi))

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(model)
