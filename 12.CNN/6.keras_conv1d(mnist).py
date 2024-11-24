# [MXDL-12-02] 6.keras_conv1d(mnist).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/NHY6y3UWvwQ
#
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.layers import Activation, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Read an MNIST dataset
with open('data/mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)

x = x.reshape(-1, 28, 28) / 255
y = y.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y)
n_class = len(set(y_train.reshape(-1,)))

k_size = 5                  # kernel size
n_kernel = 20               # number of kernels
n_pool = 10					# pooling size
n_row = x_train.shape[1]    # number of rows of an image
n_col = x_train.shape[2]    # number of columns of an image

# Build a CNN model
x_input = Input(batch_shape=(None, n_row, n_col))
conv = Conv1D(n_kernel, k_size)(x_input)
conv = Activation('relu')(conv)
pool = MaxPooling1D(pool_size=n_pool, strides=1)(conv)
flat = Flatten()(pool)
y_output = Dense(n_class, activation='softmax')(flat)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam')
model.summary()

# Training
hist = model.fit(x_train, y_train, epochs=300, batch_size=1000)

# Visually see the loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1).reshape(-1,1)
acc = (y_test == y_pred).mean()
print('Accuracy of the test data ={:.4f}'.format(acc))

# Let's check out some misclassified images.
n_sample = 10
miss_cls = np.where(y_test != y_pred)[0]
miss_sam = np.random.choice(miss_cls, n_sample)

fig, ax = plt.subplots(1, n_sample, figsize=(14,4))
for i, miss in enumerate(miss_sam):
    x = x_test[miss]
    ax[i].imshow(x.reshape(28, 28))
    ax[i].axis('off')
    ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))

