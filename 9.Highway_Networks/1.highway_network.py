# [MXDL-9-01] 1.highway_network.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/10x2lZ2lEg4
#
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pickle

# Read MNIST dataset
# x, y = fetch_openml('mnist_784', return_X_y=True)
# x = np.array(x) / 255
# y = np.array(y.to_numpy().astype('int8')).reshape(-1,1)
# with open('mnist.pkl', 'wb') as f:
#  	pickle.dump([x, y], f, pickle.HIGHEST_PROTOCOL)

with open('mnist.pkl', 'rb') as f:
 	x, y = pickle.load(f)
    
x_train,x_test,y_train,y_test=train_test_split(x, y)
n_class = len(set(y_train.reshape(-1,)))

# Highway Networks
def HighWay(x, n_layers):
    for i in range(n_layers):
        H = Dense(x.shape[-1], 
                  kernel_initializer='he_normal',
                  bias_initializer='zeros',
                  activation = 'relu')(x)
        
        T = Dense(x.shape[-1],
              kernel_initializer='glorot_normal',
              bias_initializer = 'zeros',
              activation = 'sigmoid')(x)                          
    x = H * T + x * (1. - T)
    return x

# Create an ANN model with Highway networks
x_input = Input(batch_shape=(None, x_train.shape[1]))
h = HighWay(x_input, n_layers = 20)
y_output = Dense(n_class, activation='softmax')(h)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam')
model.summary()

# Training
hist = model.fit(x_train, y_train, 
                 batch_size=1000, 
                 epochs=50, 
                 validation_data = (x_test, y_test))

# Visually see the loss history
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Test loss')
plt.legend()
plt.title("Loss history")
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
