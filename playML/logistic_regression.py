# -*- coding: utf-8 -*-
import numpy as np
from .metrics import accuracy_score


class LogisticRegression(object):
    def __init__(self):
        """初始化逻辑回归模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train，y_train，使用梯度下降法训练逻辑回归模型"""
        assert X_train.shape[0] == y_train.shape[0], '训练集与结果集的样本数必须一致'

        def J(theta, X_b, y):
            """定义损失函数"""
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return np.sum(np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat))) / -len(y)
            except:
                return float('inf')

        def derivative_J(theta, X_b, y):
            """求逻辑回归的梯度"""
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """梯度下降法求θ"""
            theta = initial_theta
            iters = 0
            while iters < n_iters:
                gradient = derivative_J(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                iters += 1
            return theta

        X_b = np.hstack((np.ones((len(X_train), 1)), X_train))
        initial_theta = np.zeros(X_b.shape[1])  # 初始的θ向量都是0
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        X_b = np.hstack([np.ones(shape=(X_predict.shape[0], 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        proba = self.predict_proba(X_predict)
        return np.array(proba >= .5, dtype=int) # 把True/False的向量转化为1，0的向量

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return 'LogisticRegression()'