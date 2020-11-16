"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
"""

import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder
from collections import Counter

ucr = UCR_UEA_datasets()
# UCR/UEA univariate and multivariate datasets.
all_ucr_datasets = ucr.list_datasets()

fixlength=32

def load_casas(dataset):
    X = np.load('./npy/{}-x.npy'.format(dataset), allow_pickle=True); X=X[:-1*(X.shape[0]%fixlength)].reshape(-1, 32, 1); print(X.shape)
    Y = np.load('./npy/{}-y.npy'.format(dataset), allow_pickle=True); Y=Y[:-1*(Y.shape[0]%fixlength)].reshape(-1, 32, 1); print(Y.shape); Y=np.array(Y, dtype=int)

    y=[]
    for i in range(X.shape[0]):
      y.append(np.argmax(np.bincount(Y[i].flatten())))
    print(Counter(y))
    X=np.array(X, dtype=object)
    y=np.array(y, dtype=object); y = y.reshape(-1,1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    dictActivities = np.load('./npy/{}-labels.npy'.format(dataset), allow_pickle=True).item()

    X_scaled=TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y, dictActivities


def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y


def load_data(dataset_name):
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)
