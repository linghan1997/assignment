from math import exp, sqrt
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from builtins import object
import numpy as np


class RBF(object):
    def __init__(self, input_dim, num_centers, out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim  # out_dim = 1
        self.centers = np.random.uniform(-1, 1, (num_centers, input_dim))
        self.sigma = 0.5
        self.W = np.random.random((self.num_centers, self.out_dim))  # bias: +1

    # Method1: Random Selection: fix sigma and centers, only learn weights W
    def train_RS(self, X, y):
        # 打乱idx后选取前num_centers个idx，并取出对应的样本向量存于centers
        center_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = X[center_idx, :]

        # sigma = dmax / sqrt(2 * num_centers)
        self.sigma = np.max(pairwise_distances(self.centers)) / (sqrt(2 * self.num_centers))

        G = self.cal_act(X)
        # learn weights by Normal Equation
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), y)  # y is a col. vector

    # Method2: Prototypes of training samples as neuron centers
    def train_PT(self, X, y):
        # using K-means to find prototypes from the training set
        k_means = KMeans(n_clusters=self.num_centers).fit(X)
        self.centers = k_means.cluster_centers_

        self.sigma = np.max(pairwise_distances(self.centers)) / (sqrt(2 * self.num_centers))
        G = self.cal_act(X)
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), y)

    # Method3: Center Selection as a model selection problem
    def train_CS(self, X, y):
        self.sigma = 0.75
        # regard the training set as candidates set
        candidates = X
        # bottom-up method
        cur_centers = 1
        # 初始化一个0向量
        self.centers = np.zeros((1, self.input_dim))
        while cur_centers <= self.num_centers:
            # 初始化accuracy数组
            acr = []
            for i in range(len(candidates)):
                self.centers[cur_centers - 1, :] = candidates[i, :]
                G = self.cal_act(X)
                self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), y)  # it might be singular matrices
                y_predict = self.predict(X)
                acr.append(self.cal_accuracy(y_predict, y))
            # 找到准确率最高的candidate
            idx = np.argmax(acr)
            # 将其放入center vectors
            self.centers[cur_centers - 1, :] = candidates[idx, :]
            # 监控过程
            print("current neurons size: ", self.centers.shape)
            # 添加一个0向量，下次循环被新的candidate覆盖
            self.centers = np.append(self.centers, np.zeros((1, self.input_dim)), axis=0)
            # 将已被选出的candidate从候选池中删除
            candidates = np.delete(candidates, idx, axis=0)
            # 下次循环模型的#neurons ++
            cur_centers += 1
        # 将最后一次循环添加的0向量删除
        self.centers = np.delete(self.centers, -1, axis=0)
        # 使用更新后的centers计算G和W
        G = self.cal_act(X)
        self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), y)

    # Method4: Backpropagation of W, sigma, c
    def train_BP(self, X, y, iter_num, eta1, eta2, eta3):
        for i in range(iter_num):
            y_predict = self.predict(X)
            # weight estimation
            dW = self.weight_estimation(X, y_predict, y)
            # center location estimation
            dC = self.center_estimation(X, y_predict, y)
            # sigma estimation
            dS = self.sigma_estimation(X, y_predict, y)
            # update
            self.W -= eta1 * dW
            self.centers -= eta2 * dC
            self.sigma -= eta3 * dS

    # 计算激活矩阵G，把每个样本的激活值排列为row vector
    def cal_act(self, X):
        # 初始化G，num_samples * num_centers
        # Gij = Oi(j) = basis_func(sample[j], center[i])
        G = np.ones((X.shape[0], self.centers.shape[0]), dtype=np.float)   # method3中G的大小会在迭代中变化
        for c_idx, c_val in enumerate(self.centers):
            for x_idx, x_val in enumerate(X):
                G[x_idx, c_idx] = self.basis_func(x_val, c_val)
        return G

    # 神经元激活公式 Gaussian（distance, sigma)
    def basis_func(self, x, c):
        return np.exp(- (norm(x - c) ** 2) / (2 * self.sigma ** 2))

    # 预测阶段：将标签调整为+-1
    def predict(self, X):
        G = self.cal_act(X)
        y = np.dot(G, self.W)
        y[y >= 0] = 1
        y[y <= 0] = -1
        return y

    def cal_accuracy(self, Y, Y_g):
        res = len(np.where((Y - Y_g) == 0)[0]) / len(Y_g)
        return res

    # tools of BP
    # Weight Estimation
    def weight_estimation(self, X, y, y_predict):
        # # W为num_centers * 1
        # dW = np.zeros((self.num_centers, 1), float)
        # # 计算每个神经元的dW
        # for idx, val in enumerate(self.centers):
        #     temp = np.exp(-(1 / (2 * self.sigma ** 2)) * cdist(val.reshape(1, -1), X) ** 2)
        #     err = y_predict - y
        #     dW[idx, :] = np.dot(temp, err)

        # Vectorized
        # N = # samples, m = # neurons
        distance = cdist(X, self.centers)  # distance为N*m
        temp = np.power(distance, 2)
        temp = np.exp(- temp * (1 / (2 * (self.sigma ** 2))))
        # err为N*1
        err = y_predict - y
        dW = np.dot(temp.T, err)
        return dW

    # Center Location Estimation
    def center_estimation(self, X, Y_predict, Y):
        dEdc = np.zeros((self.num_centers, X.shape[1]), float)
        for ci, c_val in enumerate(self.centers):
            dEdc[ci, :] = np.dot(self.W[ci, :],
                                 np.dot(((Y_predict - Y) * self.basisfunc1(c_val.reshape(1, -1), X)).T,
                                        ((X - c_val) / (2 * self.sigma ** 2))))
        return dEdc

    # Width Estimation
    def sigma_estimation(self, X, y, y_predict):
        G = np.zeros((X.shape[0], self.num_centers), float)
        for ci, c_val in enumerate(self.centers):
            for xi, x_val in enumerate(X):
                G[xi, ci] = self.basisfunc2(c_val, x_val)
        dS = np.sum(((y_predict - y) * np.dot(G, self.W)))
        return dS

    def basisfunc1(self, C, D):
        assert D.shape[1] == self.input_dim
        return np.exp(-(1 / (2 * self.sigma ** 2)) * cdist(D, C) ** 2)

    def basisfunc2(self, C, D):
        return exp(-(1 / (2 * self.sigma ** 2)) * norm(C - D) ** 2) * (norm(C - D) ** 2) / (self.sigma ** 3)