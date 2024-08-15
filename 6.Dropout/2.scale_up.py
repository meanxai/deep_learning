# [MXDL-6-02] 2.scale_up.py
# Check the scale-up results of dropout in Keras.
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/OrxJmX4WHTA
#
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
import numpy as np

x = np.array([[1, 1]], dtype=np.float32)   # input
y = np.array([[0.01]], dtype=np.float32)   # desired output

x_input = Input(batch_shape=(None, 2))
h1 = Dense(4, name='h1')(x_input)
h2 = Dropout(rate = 0.5, name='h2')(h1)
y_output = Dense(1, name='y')(h2)

model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.1))
# model.summary()

model_h1 = Model(x_input, h1)
model_h2 = Model(x_input, h2)

# Training
model.fit(x, y)

# before Dropout
h1_output = model_h1(x)
print('\nbefore Dropout (h1):\n', h1_output.numpy().round(4))

# after Dropout
h2_output = model_h2(x, training=True)
print('\nafter Dropout (h2):\n', h2_output.numpy().round(4))

