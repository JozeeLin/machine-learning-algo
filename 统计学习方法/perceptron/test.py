#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
import perceptron
df = pd.read_csv('iris.csv')
df_X = df[[u'sepal_length', u'sepal_width', u'petal_length', u'petal_width']]
df_np = np.array(df_X)
model = perceptron.DualPerceptron()
model.fit(df_np,df.target)
print '使用感知机的对偶形式算法求解'
print 'feature: [5.7,2.8,4.1,1.3] ','label:',
print model.predict([5.7,2.8,4.1,1.3])
print 'feature: [ 5.,3.3,1.4,0.2] ','label:',
print model.predict([ 5.,3.3,1.4,0.2])
