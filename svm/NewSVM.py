#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class MySVM():
    def __init__(self, C=0.6, toler=0.001, n_iter=40):
        self.C = C
        self.toler = toler
        self.n_iter = n_iter

    def selectJrand(self,i, m):
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


    def clipAlpha(self, aj, H, L):
        """
        更新aj
        """
        if aj > H:
            return H
        if aj < L:
            return L
        return aj

    def fit(self, X, y):
        m, n = X.shape
        #初始化b和alphas(alpha有点类似权重值)
        self.b = 0
        self.alphas = np.mat(np.zeros(m)).T
        myiter = 0
        while myiter < self.n_iter:
            alphaPairsChanged = 0
            for i in range(m):
                #求最优间隔分类器的净输入
                output_i = self.predict_proba(X, y, i)
                error_i = output_i - float(y[i])
                #更新条件,在区间[-toler,toler]之间的，视为0,也就是对应着支持向量
                if((y[i]*error_i<-self.toler) and (self.alphas[i]<self.C)) or ((y[i]*error_i>self.toler) and (self.alphas[i]>0)):
                    #不满足kkt条件，就随机选取下标不为i的alpha，进行优化比较
                    j = self.selectJrand(i, m)
                    output_j = self.predict_proba(X, y, j)
                    error_j = output_j - float(y[j])
                    alphaIold = self.alphas[i].copy()
                    alphaJold = self.alphas[j].copy()

                    #L和H用于将alpha[j]调整到0-C之间，如果L==H，就不做任何改变直接执行continue
                    #y[i]=y[j]表示异侧，就相减，否则，相加
                    if y[i]!=y[j]:
                        L = max(0, self.alphas[j]-self.alphas[i])
                        H = min(self.C,self.C+self.alphas[j]-self.alphas[i])
                    else:
                        L = max(0, self.alphas[j]+self.alphas[i]-self.C)
                        H = min(self.C, self.alphas[j]+self.alphas[i])
                    if L == H:
                        continue

                    #eta为可变修改步长，如果eta==0,需要退出for循环的当前迭代过程(参考《统计学习方法》的序列最小最优化算法
                    eta = 2.0*X[i,:]*X[j,:].T - X[i,:]*X[i,:].T - X[j,:]*X[j,:].T
                    if eta >= 0:
                        continue

                    #更新alphas[j]
                    self.alphas[j] -= y[j]*(error_i-error_j)/eta
                    #使用辅助函数，进行调整
                    self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
                    #检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环
                    if(abs(self.alphas[j]-alphaJold)<0.00001):
                        print("j not moving enough")
                        continue
                    #否则，更新alphas[i]
                    self.alphas[i] += y[j]*y[i]*(alphaJold-self.alphas[j])

                    #更新b
                    b1 = self.b - error_i - y[i]*(self.alphas[i]-alphaIold)*X[i,:]*X[i,:].T - y[j]*(self.alphas[j]-alphaJold)*X[i,:]*X[j,:].T
                    b2 = self.b - error_j - y[i]*(self.alphas[i]-alphaIold)*X[i,:]*X[j,:].T - y[j]*(self.alphas[j]-alphaJold)*X[j,:]*X[j,:].T

                    self.b = (b1+b2)/2.0

                    alphaPairsChanged += 1
                    print("myiter: %d i: %d, pairs changed %d" % (myiter, i, alphaPairsChanged))

            if alphaPairsChanged == 0:
                myiter += 1
            else:
                myiter = 0
            print("iteration number: %d" % myiter)

        self.w = self.calcWs(X,y)
        return self

    def predict_proba(self, X, y, i):
        term1 = np.multiply(self.alphas, y).T
        term2 = X*X[i,:].T
        return float(term1*term2)+self.b

    def calcWs(self, X, y):
        """
        基于alpha计算w值
        w: 回归系数
        """
        m,n = X.shape
        w = np.mat(np.zeros(n)).T
        for i in range(m):
            w+= np.multiply(self.alphas[i]*y[i], X[i,:].T)
        return w

    def plotSVM(self, features, labels):
        b = np.array(self.b)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(features[:,0].flatten().A[0], features[:, 1].flatten().A[0])
        x = np.arange(-1.0, 10.0, 0.1)
        y = (-b-self.w[0,0]*x)/self.w[1,0]
        ax.plot(x, y)

        #找到支持向量，并在图中标红
        for i in range(100):
            if self.alphas[i] > 0.0:
                ax.plot(features[i,0], features[i,1], 'ro')

        plt.show()

