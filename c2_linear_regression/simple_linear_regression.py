# -*- coding: utf-8 -*-
import numpy as np
from playML.metrics import r2_score

class SimpleLinearRegression1(object):
    """
    自己手写的简陋实现简单线性回归算法
    """
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """

        :param x_train:
        :param y_train:
        :return:
        """
        assert x_train.ndim == 1, 'Simple Linear Regressor can only solve single feature training data.'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 分子
        numerator = 0.0

        # 分母
        denominator = 0.0

        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean)*(y_i - y_mean)
            denominator += (x_i - x_mean)**2

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, 'Simple Linear Regressor can only solve single feature training data.'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict'
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression1()'


class SimpleLinearRegression2(object):
    """
    自己手写的简陋实现简单线性回归算法，把for循环改为向量化运算，提升效率
    """
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """

        :param x_train:
        :param y_train:
        :return:
        """
        assert x_train.ndim == 1, 'Simple Linear Regressor can only solve single feature training data.'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 分子
        numerator = (x_train - x_mean).dot(y_train - y_mean)

        # 分母
        denominator = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, 'Simple Linear Regressor can only solve single feature training data.'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict'
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """
        根据测试数据集x_test和y_test 确定当前模型的准确度
        :param x_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return 'SimpleLinearRegression2()'


class SimpleLinearRegression(SimpleLinearRegression2):
    pass