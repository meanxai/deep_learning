# [MXDL-12-05] 10.3d_mnist(resnet14).py
# 3D MNIST classification using 3D ResNet14
# Data source: https://www.kaggle.com/datasets/doleron/augmented-mnist-3d
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/tpP97EfsXco
#
from tensorflow.keras.layers import Input, Dense, Conv3D, Activation
from tensorflow.keras.layers import BatchNormalization, ZeroPadding3D
from tensorflow.keras.layers import Add, GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import h5py
import matplotlib.pyplot as plt
 
def ResBlock(x, filters, stride):
    x1 = ZeroPadding3D((1,1,1))(x)
    x1 = Conv3D(filters, (3,3,3), stride[0])(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = ZeroPadding3D((1,1,1))(x1)
    x1 = Conv3D(filters, (3,3,3), stride[1])(x1)
    x1 = BatchNormalization()(x1)

    # down sampling the input data x to match dimensions
    if stride[0] == 2:
        x2 = Conv3D(filters, (1,1,1), strides=(2,2,2))(x)
        x2 = BatchNormalization()(x2)
    else:
        x2 = x
    
    x3 = Add()([x1, x2])        # shortcut connection
    x3 = Activation('relu')(x3)
    return x3

def ResNet14(x):
    x = ZeroPadding3D((1,1,1))(x)
    x = Conv3D(16, (3,3,3), (1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = ResBlock(x, 16, (1, 1))
    x = ResBlock(x, 16, (1, 1))
    x = ResBlock(x, 32, (2, 1))
    x = ResBlock(x, 32, (1, 1))
    x = ResBlock(x, 64, (2, 1))
    x = ResBlock(x, 64, (1, 1))
    return x
    
# Read a 3D MNIST dataset.
df = h5py.File('data/3d-mnist-luiz.h5', 'r')
x_train = np.array(df["train_x"])  # (60000, 16, 16, 16, 3)
x_test  = np.array(df["test_x"])   # (10000, 16, 16, 16, 3)
y_train = np.array(df["train_y"])  # (60000,)
y_test  = np.array(df["test_y"])   # (10000,)

# Build a ResNet14 model for 3D MNIST
x_input = Input(batch_shape=(None, *x_train.shape[1:]))
h = ResNet14(x_input)
h = GlobalAveragePooling3D()(h)
y_output = Dense(10, activation='softmax')(h)

model = Model(x_input, y_output)
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate = 0.001))
model.summary()

# Training
hist = model.fit(x_train, y_train, 
                 epochs = 50, 
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
y_pred = np.argmax(y_prob, axis=1)
acc = (y_test == y_pred).mean()
print('Accuracy for the test data = {:.4f}'.format(acc))

