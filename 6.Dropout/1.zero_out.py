# [MXDL-6-01] 1.zero_out.py
# If the output of a neuron is 0, we check that the weights
# connected to this neuron are not actually updated.
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/Bu_XEPhLOBE
#
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.models import Model
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# A mask that sets the output of the third neuron in the hidden
# layer to 0.
mask = np.array([[1., 1., 0., 1.]])
            
# Create a simple network
x_input = Input(batch_shape=(None, 2))
h1 = Dense(4, name='h1', activation='sigmoid')(x_input)
h2 = Multiply()([h1, mask])   # set the output of the third neuron
                              # to 0.
y_output = Dense(1, name='y', activation='sigmoid')(h2)

model = Model(x_input, y_output)
model.compile(loss='mse', optimizer='adam')
model_h1 = Model(x_input, h1)  # Model for checking the output h1.
model_h2 = Model(x_input, h2)  # Model for checking the output h2.

print('\n# weights of h1 before training: w1')
print(model.get_layer('h1').get_weights()[0])

print('\n# weights of y before training: w2')
print(model.get_layer('y').get_weights()[0])

# Training
print('\n----- train -----')
model.fit(x, y, epochs=100, verbose=0)

# Check the output h1.
print('\n# h1 output:\n')
print(model_h1.predict(x, verbose=0))

# Check the output h2. 
# Check that the outputs of the third neuron are all 0.
print('\n# h2 output:\n')
print(model_h2.predict(x, verbose=0))

# After training, check the weights of the h1 layer.
# Compare with before training. 
# Check if the 3rd column is the same as before training.
# Check if w's in the 3rd column are not updated.
print('\n# weights of h1 after training: w1')
print(model.get_layer('h1').get_weights()[0])

# After training, check the weights of the y layer.
# Compare with before training.
# Check if w's in the 3rd column are not updated.
print('\n# weights of y after training: w2')
print(model.get_layer('y').get_weights()[0])