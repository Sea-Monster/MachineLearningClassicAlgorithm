# -*- coding: utf-8 -*-
# 度量
import numpy as np


def accuracy_score(y_true, y_predict):
    assert np.shape(y_true)[0] == np.shape(y_predict)[0], 'the size of y_true must be equal to the size of y_predict'
    return sum(y_predict == y_true) / len(y_true)
