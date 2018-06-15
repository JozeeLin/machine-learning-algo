#!/usr/bin/env python
# coding=utf-8
from crf import *
from collections import defaultdict
import re
import sys

word_data = []
label_data = []
all_labels = set()
word_sets = defaultdict(set)
obsrvs = set()  # 观测集合
print 'preprocess training data'
for line in open(sys.argv[1]):  # 读入训练数据
    words, labels = [],[]  # 对每一行句子进行处理,分词成词语列表,以及词性标签列表
    for token in line.strip().split():  # 分词
        word,label = token.split('/')  # 获取词语及其对应的词性标签
        all_labels.add(label)  # 词性标签的集合
        word_sets[label].add(word.lower())  # 分类,按照词性标签对所有的词语进行分类
        obsrvs.add(word.lower())  # 观测集合
        words.append(word)  # 词袋
        labels.append(label)  # 序列数据的词性标签列表

    word_data.append(words)  # 元素是每一行训练数据的所有词语列表
    label_data.append(labels)

if __name__ == '__main__':
    labels = list(all_labels)
    lbls = [START] + labels + [END]  # lbls表示标记序列;START END都来自crf文件;参照条件随机场的矩阵形式P198
    # 转移特征函数 P196 t_k
    transition_functions = [
        lambda yp, y, x_v, i, _yp = _yp,_y=_y: 1 if yp==_yp and y==_y else 0 for _yp in lbls[:-1] for _y in lbls[1:]
    ]
    #print transition_functions

    def set_membership(tag):
        # 特征函数? P197
        def fun(yp,y,x_v,i):
            if i<len(x_v) and x_v[i].lower() in word_sets[tag]:
                return 1
            else:
                return 0
        return fun

    observation_functions = [set_membership(t) for t in word_sets]  # 观测序列
    misc_functions = [
        lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[^0-9a-zA-Z]+$', x_v[i]) else 0,
        lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[A-Z\.]+$',x_v[i]) else 0,
        lambda yp,y,x_v,i: 1 if i < len(x_v) and re.match('^[0-9\.]+s',x_v[i]) else 0,
    ]

    tagval_functions = [
        lambda yp,y,x_v,i,_y=_y,_x=_x: 1 if i < len(x_v) and y==_y and x_v[i].lower()==_x else 0 for _y in labels for _x in obsrvs
    ]

    crf = CRF(labels=labels,
              feature_functions=transition_functions+tagval_functions+observation_functions+misc_functions)
    vectorised_x_vecs, vectorised_y_vecs = crf.create_vector_list(word_data, label_data)
    l = lambda theta: crf.neg_likelihood_and_deriv(vectorised_x_vecs, vectorised_y_vecs, theta)
    print 'Minimizing...'

    def print_value(theta):
        print crf.neg_likelihood_and_deriv(vectorised_x_vecs, vectorised_y_vecs, theta)

    print type(l),type(crf.theta)
    val = optimize.fmin_l_bfgs_b(l,crf.theta, callback=print_value)
    print val
    theta,_,_ = val

    crf.theta = theta
    print crf.neg_likelihood_and_deriv(vectorised_x_vecs, vectorised_y_vecs, theta)
    print
    print 'Latest:'
    for x_vec in word_data[-5:]:
        print x_vec
        print crf.predict(x_vec)



