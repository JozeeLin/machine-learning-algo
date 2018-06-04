#!/usr/bin/env python
# coding=utf-8
import operator
from math import log


def majorityCnt(classList):
    '''
    多数投票
    param: classList类列表
    return: 出现次数最多的类名称
    '''
    classCount = {}  # 这是一个字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 对classList中类别分别统计,并进行排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 降序排列
    return sortedClassCount[0][0]


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    参数:
        dataSet: 待划分的数据集
        axis: 划分数据集的特征的维度
        value: 特征的值
    返回值:
        符合该特征的所有实例(并且自动移除掉该特征的所有数据)
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 该实例的指定特征的值为value
            reducedFeatVec = featVec[:axis]  # 删掉这一维特征
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)  # 这就是该子结点中的一个实例
    return retDataSet


def calcShannonEnt(dataSet):
    '''
    参数:
        dataSet:数据集
    返回值:
        数据集对应的熵
    '''
    numEntries = len(dataSet)  # 实例的个数
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个实例,统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 计算熵,并累计

    return shannonEnt


def calcConditionalEntropy(dataSet, featIndex, uniqueVals):
    '''
    计算给定特征的条件熵
    参数:
        dataSet: 数据集
        featIndex: 给定特征对应的索引号
        uniqueVals: 给定特征取值的集合
    返回值:
        条件熵
    '''
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, featIndex, value)
        prob = len(subDataSet)/float(len(dataSet))
        ce += prob * calcShannonEnt(subDataSet)
    return ce


def calcInformationGain(dataSet, baseEntropy, featIndex):
    '''
    计算信息增益
    参数:
        dataSet: 数据集
        baseEntropy: 数据集的信息熵
        featIndex: 给定特征对应的索引号
    返回值:
        特征i对数据集的信息增益
    '''
    featList = [example[featIndex] for example in dataSet]  # 获取指定特征对应的数据
    uniqueVals = set(featList)
    ce = calcConditionalEntropy(dataSet, featIndex, uniqueVals)  # 计算条件熵
    infoGain = baseEntropy - ce
    return infoGain


def calcInformationGainRate(dataSet, baseEntropy, featIndex):
    '''
    计算信息增益比
    参数:
        dataSet: 数据集
        baseEntropy: 数据集的信息熵
        featIndex: 给定特征对应的索引号
    返回值:
        信息增益比
    '''
    return calcInformationGain(dataSet, baseEntropy, featIndex)/baseEntropy


def chooseBestFeatureToSplitByID3(dataSet):
    '''
    选择最好的数据集划分方式:
    参数:
        dataSet
    返回值:
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i)
        if (infoGain > bestInfoGain):  # 选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度


def chooseBestFeatureToSplitByC45(dataSet):
    '''
    特征选择算法C4.5
    参数:
        dataSet: 数据集
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGainRate = calcInformationGainRate(dataSet, baseEntropy, i)
        if (infoGainRate > bestInfoGainRate):
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature


def createTree(dataSet, columns, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    '''
    创建决策树
    参数:
        dataSet: 数据集
        labels:  特征名称列表
        chooseBestFeatureToSplitFunc:  选择特征的算法,默认是ID3
    返回值:
        决策树
    '''
    classList = [example[-1] for example in dataSet]  # 数字标签列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 所有的实例类别都一样
    if len(dataSet[0]) == 1:  # 当只有一个特征的时候,遍历完所有实例返回出现次数最多的类别
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplitFunc(dataSet)  # 使用什么来作为特征选择的准则ID3
    bestFeatLabel = columns[bestFeat]
    myTree = {bestFeatLabel: {}}  # 用字典来表示树结构
    del (columns[bestFeat])  # 删除被用过的特征名称
    featValues = [example[bestFeat] for example in dataSet]  # 获取数据集中该特征的所有特征值
    uniqueVals = set(featValues)  # 特征的唯一取值
    # 递归生成当前结点下的子树
    for value in uniqueVals:
        subLabels = columns[:]  # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree


def createDataSet():
    '''
    创建数据集
    '''
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ]
    columns = [u'年龄', u'有工作', u'有房子', u'信贷情况']

    return dataSet, columns



