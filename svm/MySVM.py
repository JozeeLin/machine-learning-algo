#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i: alpha的下标
        m: 所有alpha的数目
    Returns:
        j: 返回一个不为i的随机数，在0到m之间的整数
    """
    j = i
    while j==i:
        j = int(np.random.uniform(0,m))

    return j


def clipAlpha(aj, H, L):
    """
    更新aj
    使L<=aj<=H
    Args:
        aj: 目标值
        H: 最大值
        L: 最小值
    Returns:
        aj: 目标值
    """
    if aj > H:
        return H
    if aj < L:
        return L
    return aj

def smoSimple(X, y, C, toler, n_iter):
    """
    Args:
        X: 输入特征数据集
        y: 输入标签数据集
        C: 松弛变量(常量值)
        toler: 容错率(是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率)
        n_iter: 最大迭代次数
    Returns:
        b: 模型的常量值
        alphas 拉格朗日乘数
    """
    m, n = X.shape
    #初始化b和alphas(alpha有点类似权重值)
    b = 0
    alphas = np.mat(np.zeros(m)).T
    myiter = 0
    while myiter < n_iter:
        alphaPairsChanged = 0
        for i in range(m):
            #求最优间隔分类器的净输入
            term1 = np.multiply(alphas, y).T
            term2 = X*X[i,:].T
            output_i = float(term1*term2)+b
            error_i = output_i - float(y[i])
            '''
            选定2个alpha来进行更新优化
            '''
            #更新条件,在区间[-toler,toler]之间的，视为0,也就是对应着支持向量
            if((y[i]*error_i<-toler) and (alphas[i]<C)) or ((y[i]*error_i>toler) and (alphas[i]>0)):
                #不满足kkt条件，就随机选取下标不为i的alpha，进行优化比较
                j = selectJrand(i, m)
                term1 = np.multiply(alphas, y).T
                term2 = X*X[j,:].T
                output_j = float(term1*term2)+b
                error_j = output_j - float(y[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                #L和H用于将alpha[j]调整到0-C之间，如果L==H，就不做任何改变直接执行continue
                #y[i]=y[j]表示异侧，就相减，否则，相加
                if(y[i]!=y[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    #相同不需要优化
                    print("L==H")
                    continue

                #eta为可变修改步长，如果eta==0,需要退出for循环的当前迭代过程(参考《统计学习方法》的序列最小最优化算法
                eta = 2.0*X[i,:]*X[j,:].T - X[i,:]*X[i,:].T - X[j,:]*X[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                #更新alphas[j]
                alphas[j] -= y[j]*(error_i-error_j)/eta
                #使用辅助函数，进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                #检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                #否则，更新alphas[i]
                alphas[i] += y[j]*y[i]*(alphaJold-alphas[j])

                #更新b
                b1 = b - error_i - y[i]*(alphas[i]-alphaIold)*X[i,:]*X[i,:].T - y[j]*(alphas[j]-alphaJold)*X[i,:]*X[j,:].T
                b2 = b - error_j - y[i]*(alphas[i]-alphaIold)*X[i,:]*X[j,:].T - y[j]*(alphas[j]-alphaJold)*X[j,:]*X[j,:].T

                #if(0<alphas[i]) and (C>alphas[i]):
                #    b = b1
                #elif (0<alphas[j]) and (C>alphas[j]):
                #    b = b2
                #else:
                #    b = (b1+b2)/2.0
                b = (b1+b2)/2.0

                alphaPairsChanged += 1
                print("myiter: %d i: %d, pairs changed %d" % (myiter, i, alphaPairsChanged))

        if alphaPairsChanged == 0:
            myiter += 1
        else:
            myiter = 0
        print("iteration number: %d" % myiter)
    return b, alphas

def calcWs(alphas, X, y):
    """
    基于alpha计算w值
    Args:
        alphas: 拉格朗日乘数
        X: 输入特征向量数据集
        y: 标签数据集
    Returns:
        w: 回归系数
    """
    m,n = X.shape
    w = np.mat(np.zeros(n)).T
    for i in range(m):
        w+= np.multiply(alphas[i]*y[i], X[i,:].T)
    return w

def plotfig_SVM(features, labels, ws, b, alphas):
    b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(features[:,0].flatten().A[0], features[:, 1].flatten().A[0])
    x = np.arange(-1.0, 10.0, 0.1)
    y = (-b-ws[0,0]*x)/ws[1,0]
    ax.plot(x, y)

    #找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(features[i,0], features[i,1], 'ro')

    plt.show()


if __name__ == "__main__":
    data = pd.read_table('data/testSet.txt', header=None)
    X = np.mat(data[[0,1]])
    y = np.mat(data[[2]])
    b, alphas = smoSimple(X, y, 0.6, 0.001, 40)
    print('=====================================================')
    print 'b=', b
    print 'alphas[alphas>0]=', alphas[alphas>0]
    print 'shape(alphas[alphas>0])=',alphas[alphas>0].shape
    for i in range(100):
        if alphas[i] > 0:
            print X[i], y[i]

    ws = calcWs(alphas, X, y)
    plotfig_SVM(X, y, ws, b, alphas)
