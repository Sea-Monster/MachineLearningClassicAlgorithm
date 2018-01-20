# -*- coding: utf-8 -*-
import numpy as np
from playML.metrics import r2_score


class LinearRegression(object):

    def __init__(self):
        self.coef_ = None   # 系数
        self.interception_ = None   # 截距

        self._theta = None  # 为计算方便，将来会把系数和截距合为一个θ

    def fit_normal(self, X_train:np.ndarray, y_train:np.ndarray):
        assert X_train.shape[0] == y_train.shape[0], '每一个训练样本必须对应一个标记'

        # 特征矩阵的最左列加上一个行数等于特征矩阵的由1组成的列向量
        X_b = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])

        # 正规方程解求θ
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict:np.ndarray):
        assert self.interception_ is not None and self.coef_ is not None, '评估前必须拟合'
        assert np.shape(X_predict)[1] == len(self.coef_), '要预测的样本的特征数必须与训练的样本的特征数相等'

        X_b = np.hstack([np.ones(shape=(X_predict.shape[0], 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return 'LinearRegression()'