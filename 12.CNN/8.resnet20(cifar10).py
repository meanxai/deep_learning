# [MXDL-12-04] 8.resnet20(cifar10).py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/GdlZFrItscM
#
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import Add, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

def ResBlock(x, filters, stride):
    x1 = ZeroPadding2D((1,1))(x)
    x1 = Conv2D(filters, (3,3), stride[0])(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = ZeroPadding2D((1,1))(x1)
    x1 = Conv2D(filters, (3,3), stride[1])(x1)
    x1 = BatchNormalization()(x1)

    # down sampling the input data x to match dimensions
    if stride[0] == 2:
        x2 = Conv2D(filters, (1,1), strides=(2,2))(x)
        x2 = BatchNormalization()(x2)
    else:
        x2 = x
    
    x3 = Add()([x1, x2])        # shortcut connection
    x3 = Activation('relu')(x3)
    return x3

def ResNet20(x):
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(16, (3,3), (1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = ResBlock(x, 16, (1, 1))
    x = ResBlock(x, 16, (1, 1))
    x = ResBlock(x, 16, (1, 1))
    x = ResBlock(x, 32, (2, 1))
    x = ResBlock(x, 32, (1, 1))
    x = ResBlock(x, 32, (1, 1))
    x = ResBlock(x, 64, (2, 1))
    x = ResBlock(x, 64, (1, 1))
    x = ResBlock(x, 64, (1, 1))
    return x
    
# Read a CiFAR10 dataset
with open('data/cifar10.pkl', 'rb') as f:
 	x_train, y_train, x_test, y_test = pickle.load(f)

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']

# Build a ResNet20 model for CIFAR10
x_input = Input(batch_shape=(None, *x_train.shape[1:]))
x = RandomFlip(mode='horizontal_and_vertical')(x_input)
x = RandomRotation(0.2)(x)
x = RandomZoom(0.1)(x)
h = ResNet20(x)
h = GlobalAveragePooling2D()(h)
y_output = Dense(10, activation='softmax')(h)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001))
model.summary()

# Training
hist = model.fit(x_train, y_train, 
                 epochs = 1000, 
                 batch_size = 256,
                 validation_data=(x_test, y_test),
                 shuffle = True)

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'][0:], color='blue', label='train loss')
plt.plot(hist.history['val_loss'][0:], color='red', label='test loss')
plt.legend()
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Accuracy for test data
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1).reshape(-1,1)
acc = (y_test == y_pred).mean()
print('Accuracy for the test data = {:.4f}'.format(acc))

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
