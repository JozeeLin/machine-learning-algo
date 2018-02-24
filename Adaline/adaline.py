#!/usr/bin/env python
# coding=utf-8
import numpy as np
class AdalineGD(object):
    """Adaptive Linear Neuron classifier"""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ =  np.zeros(1+X.shape[1]) #权重初始化为0
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            #更新权重
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()

            cost = (errors**2).sum()/2 #计算代价函数值
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """求假设函数"""
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        '''求激励函数，相当与假设函数'''
        return self.net_input(X)

    def predict(self, X):
        '''返回量化结果，class label'''
        return np.where(self.activation(X)>=0,1,-1)
