# [MXDL-12-06] 12.conv_lstm_m2o(train).py - Many-to-One model
# data: http://www.cs.toronto.edu/~nitish/unsupervised_video/
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/FzUAPtDgA_o
#
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization
from myConvLSTM2D import MyConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# Load a moving MNIST dataset
with open('data/mv_mnist.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test, _ = pickle.load(f)

# The last frame of the sequence
y_train = y_train[:, -1, :, :, :] # (4500, 64, 64, 1)
y_test = y_test[:, -1, :, :, :]   # ( 500, 64, 64, 1)

# Build a Convolutional LSTM model
x = Input(shape=(x_train.shape[1], *x_train.shape[2:]))
h = MyConvLSTM2D(filters=32, kernel_size=(5, 5), pad='same', d=x_train.shape)(x) 
h = BatchNormalization()(h)
h = Activation('relu')(h)
y = Conv2D(filters=1, kernel_size=(3, 3), 
		padding='same', activation='sigmoid')(h)
	
model = Model(x, y)
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.001))
model.summary()

hist = model.fit(x_train, y_train, 
                 batch_size = 50, 
                 epochs = 100,
				 validation_data=(x_test, y_test))

# Save the trained model
model.save_weights('data/conv_lstm_m2o/weights')

# Loss history
plt.figure(figsize=(5, 3))
plt.plot(hist.history['loss'], color='blue', label='train loss')
plt.plot(hist.history['val_loss'], color='red', label='test loss')
plt.legend()
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
