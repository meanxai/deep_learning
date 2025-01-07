# [MXDL-13-04] 8.creditcard(data).py
# data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
#
# This code was used in the deep learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found at
# https://youtu.be/0YvV1RRqXVs
#
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Read a credit card dataset
# class=0: normal data points
# class=1: abnormal data points
df = pd.read_csv("data/creditcard.csv")
data = np.array(df.drop(['Time'], axis=1))
x_feat = data[:, :-1]   # features
y_class = data[:, -1]   # class

# Normalization
x_mean = x_feat.mean(axis = 0)
x_std = x_feat.std(axis = 0)
x_feat = (x_feat - x_mean) / x_std

# Dimensionality reduction by PCA
pca = PCA(n_components = 2, random_state=4)
pca.fit(x_feat)
    
# Visualize dataset
def plot_2d(x, y, title):
    x_pca = pca.transform(x)
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

# Visualize full dataset
plot_2d(x_feat, y_class, "Full dataset")

# Generate training and test dataset
# The training dataset contains only normal data points.
# The test dataset contains a mix of normal and abnormal 
# data points. 
i_normal = np.where(y_class==0)[0]
i_abnormal = np.where(y_class==1)[0]
normal = np.take(data, i_normal, axis=0)
abnormal = np.take(data, i_abnormal, axis=0)

np.random.shuffle(normal)
n = int(i_normal.shape[0] * 0.9)   # 90%
x_train = normal[:n, :-1]
y_train = normal[:n, -1]
x_test = np.vstack([normal[n:, :-1], abnormal[:, :-1]])
y_test = np.hstack([normal[n:, -1], abnormal[:, -1]])

# Normalize
x_mean = x_train.mean(axis = 0)
x_std = x_train.std(axis = 0)
nx_train = (x_train - x_mean) / x_std
nx_test = (x_test - x_mean) / x_std

# Visualize training dataset
plot_2d(nx_train, y_train, "Train dataset")

# Visualize test dataset
plot_2d(nx_test, y_test, "Test dataset")

# Save the training and test dataset
with open('data/creditcard.pkl', 'wb') as f:
	pickle.dump([nx_train, y_train, nx_test, y_test, pca], f)
