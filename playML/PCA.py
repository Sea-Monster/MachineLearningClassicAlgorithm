# -*- coding: utf-8 -*-
import numpy as np


class PCA(object):
    def __init__(self, n_components):
        assert n_components >= 1, "n_components必须大于等于1"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """
        获得数据集X的前n个主成分
        :param X:
        :param eta:
        :param n_iters:
        :return:
        """
        assert self.n_components <= np.shape(X)[1], 'n_components must not be greater than the feature number of X'

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            """效用函数"""
            return np.sum((X.dot(w) ** 2))/len(X)

        def derivative_f(w, X):
            """求梯度"""
            return X.T.dot(X.dot(w))*2./len(X)

        def direction(w):
            return w/np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = derivative_f(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, np.shape(X)[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i,:] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1,1)*w
        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert np.shape(X)[1] == np.shape(self.components_)[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert np.shape(X)[1] == np.shape(self.components_)[0]
        return X.dot(self.components_)

    def __repr__(self):
        return 'PCA(n_components=%d)' %self.n_components
