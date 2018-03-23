#!/usr/bin/env python
# coding=utf-8
import numpy as np
class MyAdaboost():
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def stumpClassify(self, X, dimen, threshVal, threshIneq):
        """
        功能:通过跟阈值进行比较对数据进行分类，分为[1,-1]
        """
        retArray = np.ones((X.shape[0],1))
        if threshIneq == 'lt':
            retArray[X[:,dimen]<=threshVal] = -1.0
        else:
            retArray[X[:,dimen]>threshVal] = -1.0
        return retArray

    def buildStump(self, X, y, D):
        """
        遍历stumpClassify函数的所有可能输入值，并找到数据集上最佳的单层决策树
        输入为当前的权重向量，数据集，返回具有最小错误率的单层决策树，最小误差率以及估计的类别向量
        """
        dataMatrix = np.mat(X)
        labelMat = np.mat(y).T
        m,n = dataMatrix.shape
        numSteps = 10.0 #用于在特征的所有可能值上进行遍历
        bestStump = {} #用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
        bestClasEst = np.mat(np.zeros((m,1)))
        minError = np.inf #存放当前最小错误率
        for i in range(n): #遍历所有特征
            rangeMin = dataMatrix[:,i].min()
            rangeMax = dataMatrix[:,i].max()
            stepSize = (rangeMax-rangeMin)/numSteps #设置步长
            for j in range(-1, int(numSteps)+1):
                for inequal in ['lt','gt']:
                    threshVal = (rangeMin+float(j)*stepSize) #设置阈值
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal, inequal)
                    errArr = np.mat(np.ones((m,1)))
                    errArr[predictedVals == labelMat] = 0

                    weightedError = D.T*errArr #计算加权错误率，也就是分类误差率

                    if weightedError < minError:
                        #当前分类误差率为最小误差率，所以在bestStump中保存该单层决策树
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump,minError, bestClasEst

    def fit(self, X, y):
        weakClassArr = []
        m = X.shape[0]
        D = np.mat(np.ones((m,1))/m)
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(self.n_iter):
            bestStump,error,classEst = self.buildStump(X, y, D)
            alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16))) #计算单层决策树的系数，确保不会发生除0溢出
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            expon = np.multiply(-alpha*np.mat(y).T, classEst) #指数项
            D = np.multiply(D, np.exp(expon)) #权重更新
            D = D/D.sum() #权重除以规范化因子，进行归一处理
            aggClassEst += alpha*classEst #弱分类器的线性组合
            aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(y).T, np.ones((m,1))) #误分类样本数
            errorRate = aggErrors.sum()/m #误分类率
            if errorRate == 0.0:break #全部被正确分类

        self.weakClass_ = weakClassArr
        self.aggClassEst_ = aggClassEst
        return self

    def predict(self, X):
        dataMatrix = np.mat(X)
        m = dataMatrix.shape[0]
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(len(self.weakClass_)):
            classEst = self.stumpClassify(dataMatrix, self.weakClass_[i]['dim'],
                                          self.weakClass_[i]['thresh'],
                                          self.weakClass_[i]['ineq'])
            aggClassEst += self.weakClass_[i]['alpha']*classEst
            aggClassEst += self.weakClass_[i]['alpha']*classEst
        return np.sign(aggClassEst) #sign函数就是参数大于0返回1，小于0返回-1，等于0返回0

