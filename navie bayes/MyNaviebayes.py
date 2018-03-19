#!/usr/bin/env python
# coding=utf-8
import numpy as np

#朴素贝叶斯
class MyNavieBayes():
    def __init__(self):
        pass

    #开始实现算法，从词向量计算概率
    def fit(self, trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        #先验概率
        pAbusive = sum(trainCategory)/float(numTrainDocs) #正样本占总样本的比例

        #似然度计算
        #p0Num = np.zeros(numWords) #分别统计不同类别中各个单词的出现次数
        #修正，初始化为全1数组
        p0Num = np.ones(numWords)
        #修正，初始化为全1数组
        p1Num = np.ones(numWords)

        #p0Denom = 0.0 #分别统计不同类别中所有的单词出现的总次数
        #修正，初始化为2
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        #为了防止小数下溢问题
        p1Vect = np.log(p1Num/p1Denom) # 正样本中，各个词出现的频率change to log()
        p0Vect = np.log(p0Num/p0Denom) # 负样本中，各个词出现的频率change to log()

        return p0Vect, p1Vect, pAbusive


    def predict(self, vec2Classify, p0Vec, p1Vec, pClass1):
        #log项为先验概率，sum项为似然度，由于使用对数来求似然度，所以乘以先验概率变成加上先验概率的对数值
        p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
        p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)

        if p1 > p0:
            return 1
        else:
            return 0
