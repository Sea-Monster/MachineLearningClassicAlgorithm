# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k: int, X_train: np.ndarray, y_train: np.ndarray, x: np.ndarray):
    """
    kNN分类算法
    :param k:   kNN的k值
    :param X_train: 训练集的特征（矩阵）
    :param y_train: 训练集的标记（向量）
    :param x: 需要预测的特征（向量）
    :return:
    """
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], "训练集中，特征向量的记录数与标记的记录数目必须一致"
    assert X_train.shape[1] == x.shape[0], '需要预测的x的特征数目必须等于训练集中的特征数目'

    # 求x与每一条记录的欧拉距离
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]

    nearest = np.argsort(distances)

    # 从y_train中取前k个与x距离最近的y
    topK_y = [y_train[i] for i in nearest[:k]]

    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]


class KNNClassifier(object):
    """
    重新整理自己写的kNN算法，使他更符合scikit-Learn的模式
    """
    def __init__(self, k):
        """
        初始化kNN分类器
        :param k:
        """
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        根据训练数据集X_train和y_train训练kNN分类器
        :param X_train:
        :param y_train:
        :return:
        """
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        给定待预测数据集X_predict，返回表示X_predict的结果向量
        :param X_predict:
        :return:
        """
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        给定单个带预测数据x，返回x_predict的预测结果值
        :param x:
        :return:
        """
        # 差不多就是把kNN_classify方法的内容全部搬过来

        # 求x与每一条记录的欧拉距离
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]

        nearest = np.argsort(distances)

        # 从y_train中取前k个与x距离最近的y
        topK_y = [self._y_train[i] for i in nearest[:self.k]]

        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return 'kNN(k=%d)'%self.k