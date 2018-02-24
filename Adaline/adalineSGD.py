#!/usr/bin/env python
# coding=utf-8
from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    """Adaptive Linear Neuron classifier
    Parameters
    -------------
    eta: float
        Learning rate(between 0 and 1.0)
    n_iter: int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassifications in every epoch.
    shuffle: bool (default: True)
        Shuffles training data every epoch
        if True to prevent cycles.
    random_state: int(default:None)
        Set random state for shuffling
        and initializing the weights.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)


    def fit(self, X, y):
        """Fit training data

        Parameters
        -------------
        X:  {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y:  array-like, shape=[n_samples]
            Target values.

        Returns
        -------------
        self: object
        """

        self._initialize_weights(X.shape[1]) #初始化权重
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X, y) #把所有的样本进行打乱
            cost = []
            for xi,target in zip(X, y):
                #每次使用一个样本对权重更新
                cost.append(self._update_weights(xi, target))
            #为了检验算法在训练后是否收敛，将每次迭代后计算出的代价函数值作为训练样本的平均消耗
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights for online learning"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1: #为什么要处理
            for xi,target in zip(X, y):
                self._update_weights(xi, target)
        else:
            #only one sample
            self._update_weights(X, y) #为什么直接更新权重？
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        '''Initialize weights to zeros'''
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights'''
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] +=  self.eta * error
        cost = 0.5*error**2 #计算代价函数值
        return cost

    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        '''Compute linear activation'''
        return self.net_input(X)

    def predict(self, X):
        '''Return class label after unit step'''
        return np.where(self.activation(X) >= 0.0, 1, -1)






