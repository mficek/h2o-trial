from typing import Tuple, Union
import numpy as np
import pandas as pd
from zlib import crc32


def belongs_to_test(id: int, test_size: float) -> bool:
    '''
    Assign an id into test set when it's hash belongs to `test_size` part of int32 range.

    :param id: value of identifier
    :param test_size: test_size as ratio between 0 and 1
    '''
    return crc32(np.int64(id)) & 0xffffffff < test_size * 2 ** 32


def train_test_split(data: pd.DataFrame, test_size: float = 0.2, random_state: Union[int, None] = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split dataframe into train and test set.

    Test set will be of size `test_size`*length of whole dataset.

    :param data: dataset to split
    :param test_size: size of test set as ratio between 0 and 1
    :param random_state: set random state for reproducibility, None for "random"
    '''
    if random_state:
        np.random.seed(random_state)
    ids = np.random.permutation(range(len(data)))
    in_test_set = np.array([belongs_to_test(id, test_size) for id in ids])
    return data.loc[~in_test_set].copy(), data.loc[in_test_set].copy()

def get_X_y(data):
    """
    Use last column from data as response variable `y`, the rest before is `X`, features.
    :param data:
    """
    X = np.array(data)[:, :-1]
    y = np.array(data)[:, -1]
    return X, y


class MinMaxScaler():
    def __init__(self):
        self.data_min = None
        self.data_max = None
        self._fitted = False

    def fit(self, X):
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        self._fitted = True

    def transform(self, X):
        if not self._fitted:
            raise ValueError('The MinMaxScaler must be fitted first.')
        X_std = (X - self.data_min) / (self.data_max - self.data_min)
        X_scaled = X_std * (1 - 0) + 0  # dummy, but feature range can be changed in the future
        return X_scaled



class PerformanceMetric():
    def __init__(self, prediction, truth):
        self.accuracy = np.mean((prediction == truth))
        self.TP = np.sum(np.logical_and(prediction==1, truth==1))
        self.TN = np.sum(np.logical_and(prediction == 0, truth == 0))
        self.FP = np.sum(np.logical_and(prediction == 1, truth == 0))
        self.FN = np.sum(np.logical_and(prediction == 0, truth == 1))
        self.precision = self.TP/(self.TP+self.FP)
        self.recall = self.TP/(self.TP+self.FN)
        self.f1_score = self.TP/(self.TP+(self.FN+self.FP)/2)
