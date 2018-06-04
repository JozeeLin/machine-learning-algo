#!/usr/bin/env python
# coding=utf-8
import sys
from DTree import *
import treePlotter

reload(sys)
sys.setdefaultencoding('utf-8')
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

myDat, columns = createDataSet()
myTreeID3 = createTree(myDat, columns)
myTreeC45 = createTree(myDat, columns, chooseBestFeatureToSplitByC45)

# 绘制决策树
treePlotter.createPlot(myTreeID3)
treePlotter.createPlot(myTreeC45)
