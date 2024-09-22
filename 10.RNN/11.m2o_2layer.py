# [MXDL-10-08] 11.m2o_2layer.py
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/Tzo1IlXjxyY
#
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# Read a MNIST dataset
mnist = fetch_openml('mnist_784', parser='auto')
x = np.array(mnist['data']).reshape(-1, 28, 28) / 255
y = np.array(mnist['target']).astype('int').reshape(-1,1)

# Split the dataset into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
n_step = x_train.shape[1]

# Build a 2-layered many-to-one LSTM model
n_feat = x_train.shape[-1]
n_output = len(set(y.reshape(-1,)))
n_hidden = 50

x_input = Input(batch_shape=(None, n_step, n_feat))
h1 = LSTM(n_hidden, return_sequences=True)(x_input)
h2 = LSTM(n_hidden, return_sequences=False)(h1)
y_output = Dense(n_output, activation='softmax')(h2)
model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.001))

# Training
hist = model.fit(x_train, y_train, epochs=100, batch_size=1000)

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
print('Accuracy of test data ={:.4f}'.format(acc))

# Let's check out some misclassified images.
n_sample = 10
miss_cls = np.where(y_test != y_pred)[0]
miss_sam = np.random.choice(miss_cls, n_sample)
fig, ax = plt.subplots(1, n_sample, figsize=(14,4))
for i, miss in enumerate(miss_sam):
    x = x_test[miss] * 255
    ax[i].imshow(x.reshape(28, 28))
    ax[i].axis('off')
    ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))

