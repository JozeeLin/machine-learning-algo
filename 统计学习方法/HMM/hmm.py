#!/usr/bin/env python
# coding=utf-8

import numpy as np


class HMM(object):
    '''
    一阶HMM模型
    Order 1 Hidden Markov Model
    参数:
        A: 状态转移概率矩阵
        B: 观测概率矩阵
        pi: 初始状态向量
    '''
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def simulate(self, T):
        def draw_from(probs):
            return np.where(np.random.multinomial(1,probs)==1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        # 根据self.pi状态分布,生成第一个状态
        states[0] = draw_from(self.pi)
        # 有了状态, 去除状态对应的观测的概率分布,生成一个观测
        observations[0] = draw_from(self.B[states[0],:])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1], :])
            observations[t] = draw_from(self.B[states[t],:])
        return observations, states

    def _forward(self, obs_seq):
        N = self.A.shape[0]  # 一共有多少种状态
        T = len(obs_seq)  # 状态序列长度

        # 求alpha->F
        F = np.zeros((N, T))  # 状态转移矩阵
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]  # 第一个状态对应的概率向量

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:,t-1], (self.A[:,n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N,T))
        X[:, -1:] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                X[n,t] = np.sum(X[:, t+1]*self.A[n,:]*self.B[:, obs_seq[t+1]])

        return X

    def observation_prob(self, obs_seq):
        '''
        求给定马尔可夫模型的条件下,预测序列的条件概率
        '''
        return np.sum(self._forward(obs_seq)[:, -1])

    def state_path(self, obs_seq):
        '''
        使用维特比算法求最优状态路径
        '''
        V,prev = self.viterbi(obs_seq)
        last_state = np.argmax(V[:,-1])  # np.argmax
        path = list(self.build_viterbi_path(prev, last_state))  # 从后向前找到最优路径
        return V[last_state,-1],reversed(path)

    def viterbi(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T-1, N), dtype=int)

        V = np.zeros((N,T))
        V[:,0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1,T):
            for n in range(N):
                seq_probs = V[:, t-1] * self.A[:, n] * self.B[n, obs_seq[t]]
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)
        return V,prev

    def build_viterbi_path(self, prev, last_state):
        T = len(prev)
        yield(last_state)
        for i in range(T-1,-1,-1):
            yield(prev[i,last_state])
            last_state = prev[i, last_state]

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            alpha = self._forward(observations)
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states,n_samples-1))
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A)*self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(n_states):
                    numer = alpha[i,t]*self.A[i,:]*self.B[:,observations[t+1]].T*beta[:, t+1].T
                    xi[i,:,t] = numer/denom

            gamma = np.squeeze(np.sum(xi,axis=1))
            prod = (alpha[:, n_samples-1]*beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma, prod/np.sum(prod)))

            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1], axis=1).reshape((-1,1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

            if np.max(abs(self.pi-newpi)) < criterion and \
                    np.max(abs(self.A-newA)) < criterion and \
                    np.max(abs(self.B-newB)) < criterion:
                done = 1
            self.A[:],self.B[:],self.pi[:] = newA, newB,newpi


