#!/usr/bin/env python
# coding=utf-8
import numpy as np
#原始形式
class perceptron(object):
    '''实现感知机的原始形式,使用随机梯度下降算法进行求解.'''
    def __init__(self,eta=0.001,max_iter=5000):
        self.eta = eta
        self.max_iter=max_iter

    def fit(self, X,y):
        '''输入特征以及标签向量'''
        m,n = X.shape
        self._W = np.zeros(n+1)
        dataset_size = m
        #随机选取一个被错误分类的样本点
        for i in range(self.max_iter):
            count = 0
            for xi,label in zip(X,y):
                self._error = label*(np.dot(xi,self._W[1:])+self._W[0])
                if self._error <= 0:
                    update = self.eta*label
                    self._W[1:] += update*xi
                    self._W[0] += update
                    break
                count += 1
            if count == dataset_size:
                break

    def predict(self,x):
        return np.where((np.dot(x,self._W[1:])+self._W[0])>=0.0,1,-1)

class DualPerceptron(object):
    def __init__(self, eta=0.01,max_iter=5000):
        self.eta = eta
        self.max_iter = max_iter

    def fit(self, X,y):
        self._alpha = np.zeros(X.shape[0])
        self._b = 0
        for  i in range(self.max_iter):
            count = 0
            for index, xi in enumerate(X):
                first_term = sum([alpha*yj*xj for xj ,yj,alpha in zip(X,y,self._alpha)])
                second_term = np.dot(first_term,xi)
                self._error = y[index]*(second_term+self._b)
                if self._error <= 0.0:
                    self._alpha[index]+=self.eta
                    self._b += self.eta*y[index]
                    count += 1
            if count == 0:
                break
        #训练结束,求出W和b
        self._W = np.zeros(X.shape[1]+1)
        self._W[0] = self._b
        self._W[1:] = sum([alpha*label*xi for alpha,label,xi in zip(
            self._alpha,y,X)])

    def predict(self,x):
        return np.where((np.dot(x,self._W[1:])+self._W[0])>=0.0,1,-1)



