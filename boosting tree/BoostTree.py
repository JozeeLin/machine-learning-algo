#!/usr/bin/env python
# coding=utf-8

import numpy as np

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    单层决策树:一个树桩,只能利用一个维度的特征进行分类
    参数:
        dataMatrix: 数据
        dimen: 特征的下标
        threshVal: 阈值
        threshIneq: 大于或小于
    返回值:
        分类结果
    '''
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    构建决策树(一个树桩)
    参数:
        dataArr: 数据特征矩阵
        classLabels: 标签向量
        D: 训练数据的权重向量
    返回值:
        最佳决策树,最小的错误率加权和,最优预测结果
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf  # 将错误率之和设为正无穷
    for i in range(n):  # 遍历所有的特征
        rangeMin = dataMatrix[:, i].min()  # 该维的最小最大值
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps

        for j in range(-1, int(numSteps)+1):  # 遍历这个区间
            for inequal in ['lt', 'gt']:  # 遍历大于和小于
                threshVal = (rangeMin+float(j)*stepSize)
                # 使用参数i,j,lessThan 调用树桩决策树分类
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测正确的样本对应的错误率为0,否则为1

                weightedError = D.T * errArr  # 计算错误率加权和,对应着e_m这个公式

                if weightedError < minError:  # 记录最优树桩决策树分类器
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClasEst



def AdaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的ada训练
    参数:
        dataArr: 样本特征矩阵
        classLabels: 样本分类向量
        numIt: 迭代次数
    返回值:
        一系列弱分类器及其权重,样本分类结果
    '''
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.mat(np.ones((m, 1)) / m)  # 将每个样本的权重初始化为均等
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 获取弱分类器
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))  # 计算alpha,方式发生除数为0的错误
        bestStump['alpha'] = alpha  # 当前弱分类器的权重
        weakClassArr.append(bestStump)  # 保存树桩决策树(保存弱分类器)

        # 每个样本对应的指数,当预测值等于y的时候,恰好为-alpha,否则为alpha
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))  # 计算下一个迭代的D向量
        D = D / D.sum()   # 归一化

        # 计算所有分类器的误差,如果为0则终止训练
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error: ', errorRate
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClasify(datToClass, classifierArr):
    '''
    提升树分类
    '''
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return np.sign(aggClassEst)
