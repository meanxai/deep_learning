# [MXDL-13-01] 1.dim_reduction(1).py
# Implement an autoencoder model for dimensionality reduction, 
# and compare this with the results by principal component 
# analysis (PCA).
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/Wf8o_w1C0VM
#
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Read an MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)
     
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Build an autoencoder model for dimensionality reduction
# Encoder
x_input = Input(batch_shape=(None, x_train.shape[1]))
h_enc = Dense(300, activation='relu')(x_input)
h_enc = Dropout(0.5)(h_enc)
h_enc = Dense(100, activation='relu')(h_enc)
h_enc = Dropout(0.5)(h_enc)
z_enc = Dense(2, activation='linear',
              activity_regularizer=L2(0.01))(h_enc)

# Decoder
h_dec = Dense(100, activation='relu')(z_enc)
h_dec = Dropout(0.5)(h_dec)
h_dec = Dense(300, activation='relu')(h_dec)
h_dec = Dropout(0.5)(h_dec)
x_dec = Dense(x_train.shape[1], activation='sigmoid')(h_dec)

model = Model(x_input, x_dec)
model.compile(loss='binary_crossentropy', 
              optimizer = Adam(learning_rate=0.001))
encoder = Model(x_input, z_enc)

# training
hist = model.fit(x_train, x_train, epochs=300, batch_size=500,
                validation_data=[x_test, x_test])

# Loss history
plt.plot(hist.history['loss'], c='blue', label='train loss')
plt.plot(hist.history['val_loss'], c='red', label='test loss')
plt.legend()
plt.show()

# Reduce the dimensionality of the test data.
x_pred = encoder.predict(x_test)

# z-space
def show_image(x, y):
    plt.figure(figsize=(6,6))
    for i in np.unique(y):
        idx = np.where(y == i)[0]
        z = x[idx]
        plt.scatter(z[:, 0], z[:, 1], s=10, alpha=0.5, label=str(i))
    plt.legend()
    plt.show()

print('\nDimensionality reduction by Autoencoder', end='')
show_image(x_pred, y_test)

# Dimensionality reduction by PCA
from sklearn.decomposition import KernelPCA

x_train1 = x_train[:10000]
x_test1 = x_test[:5000]
y_test1 = y_test[:5000]

pca = KernelPCA(n_components=2)
pca.fit(x_train1)
x_pred_pca = pca.transform(x_test1)

print('\nDimensionality reduction by PCA', end='')
show_image(x_pred_pca, y_test1)

