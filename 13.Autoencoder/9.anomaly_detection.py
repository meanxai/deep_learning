# [MXDL-13-04] 9.anomaly_detection.py
# Credit card fraud detection
# data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/0YvV1RRqXVs
#
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the credit card dataset
with open('data/creditcard.pkl', 'rb') as f:
 	x_train, y_train, x_test, y_test, pca_model = pickle.load(f)

# Build an autoencoder model for anomaly detection
# Encoder
x_input = Input(batch_shape=(None, x_train.shape[1]))
h_enc = Dense(20, activation='relu')(x_input)
h_enc = Dropout(0.5)(h_enc)
z_enc = Dense(10, activation='relu')(h_enc)

# Decoder
h_dec = Dense(20, activation='relu')(z_enc)
h_dec = Dropout(0.5)(h_dec)
x_dec = Dense(x_train.shape[1])(h_dec)

model = Model(x_input, x_dec)
model.compile(loss='mean_squared_error',
              optimizer = Adam(learning_rate=0.001))
model.summary()

# Training
hist = model.fit(x_train, x_train, epochs=100, batch_size=500,
                 validation_data = [x_test, x_test])

# Loss history
plt.plot(hist.history['loss'], c='blue', label='train loss', linewidth=3)
plt.plot(hist.history['val_loss'], c='red', label='test loss', linewidth=3)
plt.legend()
plt.show()

# Prediction
x_pred = model.predict(x_test)

# We use the reconstruction error to determine whether a test data 
# point is normal or abnormal. Since the autoencoder model is trained
# with normal data, the reconstruction error for abnormal data will 
# be high. Among the test data, the n_outliers data points with high
# reconstruction error are considered anomalous.
def get_labels(x_true, x_pred, r):
    if isinstance(r, float):
        n_outliers = int(x_test.shape[0] * r)
    else:
        n_outliers = r
        
    r_err = np.sqrt(np.sum(np.square(x_true - x_pred), axis=1))
    idx = np.argsort(r_err)[-n_outliers:]
    y_pred = np.zeros(x_true.shape[0])
    y_pred[idx] = 1
    return y_pred

y_pred = get_labels(x_test, x_pred, 0.01)

# Visualize dataset
def plot_2d(x, y, title):
    x_pca = pca_model.transform(x)
    plt.figure(figsize=(6, 6))
    color = ['blue', 'red']
    label = ['normal', 'abnormal']
    for i, (c, l) in enumerate(zip(color, label)):
        p = x_pca[y == i]
        plt.scatter(p[:, 0], p[:, 1], s=20, c=c, alpha=0.5,
                    label=l + ' (' + 
                    str(p.shape[0]) + ')')
    plt.legend(fontsize=12)
    plt.title(title)
    plt.xlim(-10,30)
    plt.show()

plot_2d(x_test, y_test, "Distribution of true data")
plot_2d(x_test, y_pred, "Distribution of predicted data")

# Metrics
f_cnt = y_test[y_test==1].shape[0]
y_pred = get_labels(x_test, x_pred, f_cnt)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix:")
print(cm)
print("\n Accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("Precision: %.4f" % precision_score(y_test, y_pred))
print("   Recall: %.4f" % recall_score(y_test, y_pred))
print(" F1-score: %.4f" % f1_score(y_test, y_pred))
