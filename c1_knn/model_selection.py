# -*- coding: utf-8 -*-
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """
    将数据X和y按照test_ratio分割成X_train，X_test,y_train，y_test
    :param X:
    :param y:
    :param test_ratio:
    :param seed:
    :return:
    """
    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(np.shape(X)[0])

    test_size = int(np.shape(X)[0] * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
