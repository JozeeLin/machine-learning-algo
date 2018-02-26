import numpy as np
class myperceptron():
    """Perceptron Classifie 应用在二分类上.此代码实现了感知器算法的对偶形式"""
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1]) #将权重初始化为0或一个极小的随机数
        self.errors_ = []

        #迭代所有的训练样本
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target-self.predict(xi)) #delta wi
                self.w_[1:] += update*xi #更新所有的w
                self.w_[0] += update
                errors += int(update != 0.0) #表示xi样本被错误分类了
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)

