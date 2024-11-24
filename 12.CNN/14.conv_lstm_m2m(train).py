# [MXDL-12-06] 14.conv_lstm_m2m(train).py - Many-to-Many
# data: http://www.cs.toronto.edu/~nitish/unsupervised_video/
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/FzUAPtDgA_o
#
from tensorflow.keras.layers import Input, ConvLSTM2D, Conv3D, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Conv2D, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# Load a moving MNIST dataset
with open('data/mv_mnist.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test, _ = pickle.load(f)
 
# Build a Convolutional LSTM model
x = Input(shape=(None, *x_train.shape[2:])) # (None, None, 64, 64, 1)
h = ConvLSTM2D(filters=32, kernel_size=(5, 5), 
                padding='same', 
                return_sequences=True)(x)
h = BatchNormalization()(h)
h = Activation('relu')(h)
# y = TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3), 
#                             padding='same', 
#                             activation='sigmoid'))(h)

# Instead of using TimeDistibuted, you can use a 3D convolution 
# layer in the output layer as follows.
y = Conv3D(filters=1, kernel_size=(3,3,3), 
 		     padding='same', activation='sigmoid')(h)

model = Model(x, y)
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.001))
model.summary()

# model = load_model("data/conv_lstm_m2m.h5")

# Fit the model to the training data.
hist = model.fit(x_train, y_train, 
                 batch_size = 10, 
                 epochs = 100,
                 validation_data=(x_test, y_test))

# Save the model
model.save('data/conv_lstm_m2m_1.h5', save_format="h5")

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='blue', label='train loss')
plt.plot(hist.history['val_loss'], color='red', label='test loss')
plt.legend()
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

