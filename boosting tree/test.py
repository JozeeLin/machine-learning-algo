#!/usr/bin/env python
# coding=utf-8
import numpy as np
from BoostTree import *

def loadSimpData():
    """
加载简单数据集
    :return:
    """
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


datArr, labelArr = loadSimpData()
clasifierArr, aggClassEst = AdaBoostTrainDS(datArr, labelArr, 10)
