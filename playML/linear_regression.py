# -*- coding: utf-8 -*-
import numpy as np
from playML.metrics import r2_score
from c2_linear_regression import linear_regression

class LinearRegression(linear_regression.LinearRegression):
    def fit_gd(self, X_train:np.ndarray, y_train:np.ndarray, eta=0.01, n_iters=1e4):
        """
        根据训练数据集X_train和y_train，使用梯度下降法训练线性回归模型
        :param X_train:
        :param y_train:
        :param eta:
        :param n_iters:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], '每一个训练样本必须对应一个标记'

        def J(theta, X_b, y):
            """
            给定θ，特征矩阵X，标记向量y，根据损失函数得出其（损失）值
            :param theta:
            :param X_b:
            :param y:
            :return:
            """

            # 分子部分其实等价于 (y - X_b.dot(theta)).T.dot(y - X_b.dot(theta))
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')  # 防止溢出？有异常直接返回最大值

        def derivative_J(theta: np.ndarray, X_b: np.ndarray, y: np.ndarray):
            """
            求θ为给定值时的导数
            :param theta:
            :param X_b:
            :param y:
            :return:
            """

            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)

            # 改为向量的形式
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
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
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self