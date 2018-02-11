# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """
    计算y_true和y_predict之间的准确率
    :param y_true:
    :param y_predict:
    :return:
    """
    assert y_true.shape[0] == y_predict.shape[0], 'the size of y_true must be equal to the size of y_predict'

    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    计算y_true和y_predict之间的MSE
    :param y_true:
    :param y_predict:
    :return:
    """
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """
    计算R^2 R Square
    :param y_true:
    :param y_predict:
    :return:
    """
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'y_true与y_predict的样本数目必须一致'
    return np.sum((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'y_true与y_predict的样本数目必须一致'
    return np.sum((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'y_true与y_predict的样本数目必须一致'
    return np.sum((y_true == 1) & (y_predict == 0))


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'y_true与y_predict的样本数目必须一致'
    return np.sum((y_true == 1) & (y_predict == 1))


def confusion_matrix(y_true, y_predict):
    """返回一个2✖️2的混淆矩阵"""
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])


def precision_score(y_true, y_predict):
    """求精准率"""
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:  # 分母为0时，结果返回0
        return 0.0


def recall_score(y_true, y_predict):
    """求召回率"""
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def f1_score(y_true, y_predict):
    """f1 score"""
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)

    try:
        return 2.0 * precision * recall / (precision + recall)
    except:
        return 0.


def TPR(y_true, y_predict):
    return recall_score(y_true, y_predict)


def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.
