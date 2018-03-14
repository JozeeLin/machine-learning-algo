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

    def fit_OvR(self, X, y):
        m, n = X.shape
        labels = np.unique(y)
        num_labels = len(labels)

        self.all_theta = np.zeros((n+1, num_labels))

        #label encoding
        class_y = np.zeros((m, num_labels))
        for i in range(num_labels):
            class_y[:, i] = np.int32(y == i).reshape(1, -1)

        #使用二分类LR算法计算出对应的theta值
        for i in range(num_labels):
            self.fit(X, class_y[:, i])
            self.all_theta[:, i] = self.w_

        self.all_theta = self.all_theta.T
        return self

    def predict_OvR(self, X):
        result = predict_proba_OvR(self,X)[0]
        return np.array(result)

    def predict_proba_OvR(self, X):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, self.all_theta.T[1:, :])+self.all_theta.T[0, :])

        h_max = np.max(h, axis=1)
        result = []
        for i in np.arange(0, m):
            t = np.where(h[i, :] == h_max[i])[0].ravel()[0]
            result.append((t,h_max[i]))

        return np.array(result)
