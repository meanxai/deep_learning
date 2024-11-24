# [MXDL-12-03] 7.conv2D(cifar10).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/dOctVlgVp84
#
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import Activation, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load CiFAR10 dataset
# from tensorflow.keras.datasets import cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# with open('data/cifar10.pkl', 'wb') as f:
#  	pickle.dump([x_train, y_train, x_test, y_test], f)

with open('data/cifar10.pkl', 'rb') as f:
 	x_train, y_train, x_test, y_test = pickle.load(f)
     
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']

# Build a CNN model
def Conv2D_Pool(x, filters, k_size, p_size):
    x = Conv2D(filters, k_size)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(p_size)(x)
    return x

x_input = Input(batch_shape=(None, *x_train.shape[1:]))
x = RandomFlip(mode='horizontal')(x_input)
x = RandomRotation(0.1)(x)
h = Conv2D_Pool(x, 16, k_size=(3,3), p_size=(2,2))
h = Conv2D_Pool(h, 32, k_size=(3,3), p_size=(2,2))
h = Conv2D_Pool(h, 64, k_size=(3,3), p_size=(2,2))
h = Flatten()(h)
h = Dense(128, activation='relu')(h)
h = Dropout(0.5)(h)
y_output = Dense(10, activation='softmax')(h)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', 
	optimizer=Adam(learning_rate=0.0005))
model.summary()

# Training
hist = model.fit(x_train, y_train, 
                 epochs = 300,
                 batch_size = 1000,
                 validation_data = (x_test, y_test),
                 shuffle = True)

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='blue', label='train loss')
plt.plot(hist.history['val_loss'], color='red', label='test loss')
plt.legend()
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Accuracy for test data
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1).reshape(-1,1)
acc = (y_test == y_pred).mean()
print('Accuracy for the test data ={:.4f}'.format(acc))

# Check out some misclassified images
n_sample = 10
miss_cls = np.where(y_test != y_pred)[0]
miss_sam = np.random.choice(miss_cls, n_sample)

fig, ax = plt.subplots(1, n_sample, figsize=(14,4))
for i, miss in enumerate(miss_sam):
    ax[i].imshow(x_test[miss])
    ax[i].axis('off')
    s_test = categories[y_test[miss][0]]
    s_pred = categories[y_pred[miss][0]]
    ax[i].set_title(s_test + ' /\n' + s_pred)
plt.show()
