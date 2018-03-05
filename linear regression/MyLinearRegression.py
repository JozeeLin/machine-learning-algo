#!/usr/bin/env python
# coding=utf-8
import numpy as np
class myLinearRegression():
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        #权重初始化
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = [] #代价函数的值
        m = X.shape[0]

        #应用梯度下降算法进行权重更新
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            update = (self.eta/m)*X.T.dot(errors)
            self.w_[1:] += update #更新所有的的权重
            self.w_[0] += (self.eta/m)*errors.sum()
            cost = (errors**2).sum()/(2.0*m)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        '''假设函数'''
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        '''回归函数，直接返回假设函数的值'''
        return net_input(X)
