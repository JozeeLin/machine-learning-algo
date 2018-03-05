#!/usr/bin/env python
# coding=utf-8
import numpy as np
class myLogisticRegression():
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        #权重初始化为0时，算法性能很差，如果初始化为1,则能够完全分开所有的样本
        #self.w_ = np.ones(1+X.shape[1])
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        m = X.shape[0]

        for _ in range(self.n_iter):
            output = self.predict_proba(X)
            errors = (-y+output)
            update = (self.eta/m)*X.T.dot(errors)
            self.w_[1:] += update
            self.w_[0] += (self.eta/m)*errors.sum()
            cost = (errors**2).sum()/(2.0*m)
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict_proba(self, X):
        return self.sigmoid(self.net_input(X))

    def predict(self, X):
        return np.where(self.predict_proba(X)>0.5, 1, 0)

    def sigmoid(self, z):
        return 1/(1+np.exp(z))



