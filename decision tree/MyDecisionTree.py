#!/usr/bin/env python
# coding=utf-8
import numpy as np
import operator

class MyDTree():
    """
    reference：《统计学习方法》
    没有实现剪枝
    implement ID3 algorithm
    """
    def __init__(self):
        pass

    def calcShannonEnt(self, y):
        """
        计算经验熵
        """
        numEntries = len(y)
        uniqueLabels = y.unique()

        shannonEnt = 0.0
        for label in uniqueLabels:
            labelCount = (y==label).sum()
            prob = float(labelCount)/numEntries #求出每个分类对应的样本数占总样本数的比例
            shannonEnt -= prob*np.log2(prob) #计算所有分类包含的信息期望值
        return shannonEnt

    #按照给定特征划分数据集
    def splitDataSet(self, dataSet, axis, value):
        retDataSet = dataSet[dataSet.iloc[:,axis]==value]
        retDataSet = retDataSet.iloc[:,:axis].join(retDataSet.iloc[:,axis+1:])

        #for featVec in dataSet:
        #    if featVec[axis] == value:
        #        reducedFeatVec = featVec[:axis]
        #        reducedFeatVec.extend(featVec[axis+1:])
        #        retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet, y):
        numFeatures = dataSet.shape[1]
        baseEntropy = self.calcShannonEnt(y)
        bestInfoGain = 0.0
        if dataSet.shape[1]==0:
            print dataSet

        for i in range(numFeatures):
            uniqueVals = dataSet.iloc[:,i].unique()
            newEntropy = 0.0

            #遍历当前剩下的所有特征，分别这些特征对应的经验条件熵
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet,i,value)
                suby = y[y==value]
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(suby)

            #计算当前特征对应的信息增益
            infoGain = baseEntropy - newEntropy
            #找出信息增益最大的特征
            if(infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        #返回能够产生最大信息增益的特征索引
        return bestFeature

    #当数据集已经处理了所有属性，但是类标签依然不是唯一的，
    #此时我们需要采用多数投票的方法决定该叶子节点的分类
    def majorityCnt(self, classList):
        return classList.value_counts().index[0]

    #创建树
    def createTree(self, dataSet, y):
        #classList = [example[-1] for example in dataSet] #包含数据集的所有类标签
        #if classList.count(classList[0]) == len(classList): #所有的类标签相同，触发递归终止条件
        #    return classList[0]
        if len(y)==0:
            return None
        if (y==y.iloc[0]).sum() == len(y):
            return y.iloc[0]

        if dataSet.shape[1] == 0: #使用完了所有的标签，触发递归终止条件，按照多数投票原则，返回相应的类标签
            return self.majorityCnt(y)

        bestFeat = self.chooseBestFeatureToSplit(dataSet, y)
        bestFeatName = dataSet.columns[bestFeat] #获取特征名称
        myTree = {bestFeatName:{}}
        uniqueVals = dataSet.iloc[:,bestFeat].unique() #获取最优特征的所有可能取值
        for value in uniqueVals:
            suby = y[dataSet.iloc[:,bestFeat]==value]
            myTree[bestFeatName][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), suby)
        return myTree

    def fit(self, X, y):
        self.default_ = y.value_counts().index[0]
        self.tree = self.createTree(X,y)
        return self

    def predict(self, X):
        return np.array(self.classify(self.tree, X))

    def classify(self, inputTree, testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        firstFeat = testVec[firstStr]
        classLabels = []
        for i in range(len(testVec)):
            if firstFeat.iloc[i] not in secondDict.keys():
                classLabels.append(self.default_) #出现决策树中不存的分支时，默认设置为训练集中出现次数最多的类别

            for key in secondDict.keys():
                if firstFeat.iloc[i] == key:
                    #判断该节点是否是子节点，如果是则继续递归判断
                    if type(secondDict[key]).__name__=='dict':
                        classLabels.extend(self.classify(secondDict[key], testVec.iloc[[i]]))
                    else:
                        classLabels.append(secondDict[key])
        return classLabels
