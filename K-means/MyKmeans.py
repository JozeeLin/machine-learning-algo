#!/usr/bin/env python
# coding=utf-8
import numpy as np
class MyKmeans():
    def __init__(self, k):
        self.k = k

    def distEclud(self, vecA, vecB):
        """
        计算两个点之间的欧拉距离
        """
        return np.sqrt(np.sum(np.power(vecA-vecB, 2)))

    def randCent(self, dataSet):
        """
        随机生成起始的类重心
        参数:
            dataSet:样本数据集
            k:类重心的个数
        """
        n = dataSet.shape[1]
        centroids = np.mat(np.zeros((self.k, n))) #类重心
        for j in range(n):
            minJ = min(dataSet[:, j])
            rangeJ = float(max(dataSet[:, j])-minJ)
            #np.random.rand产生元素取值在0~1.0之间的，形状为(k,1)的数据集
            centroids[:, j] = minJ+rangeJ*np.random.rand(self.k, 1)
        return centroids

    def fit(self, X):
        """
        把训练集进行聚类，划分成k类
        """
        self.cluster_centers_, clusterAssment = self.normal_cluster(X, self.k)
        self.labels_ = clusterAssment[:, 0].A.ravel()
        return self

    def normal_cluster(self, X, k):
        #初始化类重心
        centroids = self.randCent(X)
        m = X.shape[0]
        #簇分配结果矩阵包含两列:一列记录簇索引值(也就是表示类重心的索引号),第二列存储误差
        clusterAssment = np.mat(np.zeros((m, 2)))

        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                for j in range(k):
                    #判断当前样本离那个类重心更近
                    distJI = self.distEclud(centroids[j, :], X[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i, 0] != minIndex: #样本i的类重心发生改变，只要有一个样本的类重心发生改变，都要更新类重心
                    clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
            for cent in range(k):
                ptsInClust = X[(clusterAssment[:, 0].A == cent).ravel()]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)#新类重心

        return centroids, clusterAssment


    def fit_bi(self, X):
        """
        二分K均值算法
        """
        m = X.shape[0]
        clusterAssment = np.mat(np.zeros((m, 2)))
        centroid0 = np.mean(X, axis=0).tolist()[0] #最开始只有一个簇
        centList = [centroid0] #当前所有的类重心，起始只有一个
        for j in range(m):
            clusterAssment[j, 1] = self.distEclud(np.mat(centroid0), X[j, :])**2

        while(len(centList) < self.k):
            lowestSSE = np.inf
            for i in range(len(centList)):
                #尝试划分每一个簇，对那一个簇进行划分能够最大限度的降低SSE
                ptsInCurrCluster = X[(clusterAssment[:, 0].A == i).ravel(), :]
                centroidMat, splitClusterAss = self.normal_cluster(ptsInCurrCluster, 2)
                #分别计算被进行划分的数据集对应的SSE,以及剩余下的数据集对应的SSE，两者之和为划分后总的数据集的SSE
                sseSplit = np.sum(splitClusterAss[:, 1])
                sseNotSplit = np.sum(clusterAssment[(clusterAssment[:, 0].A != i).ravel(), 1])

                #保留能够使得SSE降低幅度最大的划分方式
                if(sseSplit+sseNotSplit) < lowestSSE:
                    bestCentToSplit = i #记录当前的最优划分对应被划分的类别
                    bestNewCents = centroidMat
                    bestClustAss = splitClusterAss.copy()
                    lowestSSE = sseSplit+sseNotSplit
            bestClustAss[(bestClustAss[:, 0].A == 1).ravel(), 0] = len(centList) #对新划分到不同簇里的样本进行标注
            bestClustAss[(bestClustAss[:, 0].A == 0).ravel(), 0] = bestCentToSplit

            centList[bestCentToSplit] = bestNewCents[0, :]
            centList.append(bestNewCents[1, :])
            clusterAssment[(clusterAssment[:, 0].A == bestCentToSplit).ravel(), :] = bestClustAss

        self.labels_ = clusterAssment[:, 0].A.ravel()
        self.cluster_centers_ = centList
        return self

    def fit_predict(self,X):
        """
        把训练集的各个样本根据聚类结果进行标注
        """
        pass

    def predict(self, X):
        """
        对测试集的样本进行预测，输出他们对应的标签
        """
        pass
