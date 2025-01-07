# [MXDL-13-06] 10.VAE.py
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
latent_dim = 8
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

# Training a VAE model
model = VAE(encoder, decoder)
model.compile(optimizer=Adam())
hist = model.fit(x_train, epochs=30, batch_size=500, verbose=2)

# Loss history
plt.plot(hist.history['bce_loss'], c='blue'); plt.show()
plt.plot(hist.history['kld_loss'], c='red'); plt.show()
plt.plot(hist.history['tot_loss'], c='green'); plt.show()

# Generate multiple images similar to a test image
digit = 5
idx = np.where(y_test == digit)[0]
n = np.random.choice(idx, 1)[0]
xt = x_test[n].reshape(1, 28, 28, 1)

print("\nOriginal image:", end='')
plt.figure(figsize=(1,1))
plt.imshow(xt.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

print("\nGenerated images:", end='')
mean, log_var, _ = model.encoder.predict(xt, verbose=0)
fig, ax = plt.subplots(8, 8, figsize=(8,8))
for i in range(8):
    for k in range(8):
        z_test = Sampling()([mean, log_var]).numpy()
        y_pred = model.decoder.predict(z_test, verbose=0)
        
        ax[i, k].imshow(y_pred.reshape(28, 28), cmap='gray')
        ax[i, k].axis('off')
plt.show()

