# -*- coding: utf-8 -*-
import numpy as np


class StandardScaler(object):
    """
    照猫画虎的仿照scikit-learn实现一个Standard Scaler
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X:np.ndarray):
        """
        根据训练数据集X获得数据的均值和标准差
        (暂时只处理2维的数据)
        :param X: 
        :return: 
        """
        assert X.ndim == 2, 'The dimension of X must be 2'
        
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
        
        return self
    
    def transform(self, X):
        """
        将X根据这个StandardScaler进行0均值标准化处理
        :param X: 
        :return: 
        """
        assert X.ndim == 2, 'The dimension of X must be 2'
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform"
        X_standard = (X - self.mean_) / self.scale_
        return X_standard

    def transform_standard(self, X):
        """
        将X根据这个StandardScaler进行0均值标准化处理(老师教学版)
        :param X:
        :return:
        """
        assert X.ndim == 2, 'The dimension of X must be 2'
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform"
        X_res = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            X_res[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return X_res

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection._split import train_test_split
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=666)

    ss = StandardScaler()
    ss.fit(X_train)

    X_standard = ss.transform(X)
    print(ss.transform_standard(X))
    print('-'*100)
    print(X_standard)
