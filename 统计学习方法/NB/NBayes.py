#!/usr/bin/env python
# coding=utf-8
from math import log,exp

class LaplaceEstimate(object):
    '''
    使用拉普拉斯平滑处理的贝叶斯估计
    参见笔记中的条件贝叶斯估计公式
    '''
    def __init__(self):
        self.d = {}  # [词-词频]的映射字典
        self.total = 0.0  # 全部词的词频
        self.none = 1  # 当一个词不存在的时候,它的词频

    def exists(self, key):
        return key in self.d

    def getsum(self):
        return self.total

    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    def getprob(self, key):
        '''
        估计先验概率:
            key:词
            return:概率
        '''
        return float(self.get(key)[1])/self.total

    def samples(self):
        '''
        获取全部样本:
        '''
        return self.d.keys()

    def add(self, key, value):
        '''
        这里默认使用拉普拉斯平滑了,所以使用1来初始化频数
        '''
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


class NBayes(object):

    def __init__(self):
        self.d = {}     # [标签,概率]的字典映射
        self.total = 0  # 全部词频

    def train(self, data):
        for item in data:    # i=[[词链表],标签]
            label = item[1]  # 获取标签
            if label not in self.d:
                self.d[label] = LaplaceEstimate()  # 相当于条件贝叶斯估计
            for word in item[0]:  # 遍历词链表中的每个词
                self.d[label].add(word, 1)  # 分别对不同标签集中的词统计词频(公式中的分子)
        self.total = sum([self.d[x].getsum() for x in self.d])  # 公式中的分母
        # self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys()))

    def classify(self, x):
        tmp = {}
        for label in self.d:  # 遍历标签
            tmp[label] = log(self.d[label].getsum())-log(self.total)  # P(Y=ck)
            for word in x:
                tmp[label] += log(self.d[label].getprob(word))  # P(Xj=xj|Y=ck)
        ret, prob = 0, 0
        for c in self.d:
            now = 0
            try:
                for otherc in self.d:
                    now += exp(tmp[otherc]-tmp[c])  # 将对数还原为1/p
                now = 1/now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = c, now
        return ret, prob


class Sentiment(object):
    def __init__(self):
        self.classifier = NBayes()

    def segment(self, sent):
        words = sent.split(' ')
        return words

    def train(self, neg_docs, pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.segment(sent), u'neg'])
        for sent in pos_docs:
            data.append([self.segment(sent), u'pos'])

        self.classifier.train(data)

    def classify(self, sent):
        return self.classifier.classify(self.segment(sent))


s = Sentiment()
s.train([u'糟糕', u'好 差劲'], [u'优秀', u'很 好'])
print s.classify(u'好 优秀')
