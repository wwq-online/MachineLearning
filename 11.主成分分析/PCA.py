# -*-coding:utf-8-*-
import numpy as np
import random
import math
from numpy import *

"""
Author: wuweiqun
机器学习经典算法：PCA降维
算法步骤：
    1. 数据归一化（均值化）
    2. 计算协方差矩阵（方阵）
    3. 计算协方差矩阵的特征值与特征向量
    4. 将特征值从大到小排序
    5. 保留前K个特征值对应的特征向量
    6. 将数据转换到K个特征向量构建的空间
"""

class PCA():
    def __init__(self):
        # self,data = self.getData()
        pass

    def getData(self, dataPath='./data.txt'):
        """
        获取样本
        :param dataPath: 训练样本路径
        :return: 为训练数据T(X,Y),维度为(80, 2), 80表示样本数目; 2表示2个特征x1,x2, 数据类型为float numpy.ndarray
        """
        data = np.loadtxt(dataPath, dtype=str, encoding='utf-8', delimiter=',')
        data2fl = np.ones((data.shape[0], data.shape[1] - 1), dtype=float)

        for i in range(data.shape[0]):
            for j in range(data.shape[1] - 1):
                data2fl[i, j] = float(data[i, j])
        return data2fl

    def pca(self, data, k=1):  # topNfeat为可选参数，记录特征值个数
        # 1.数据归一化
        # 求每个维度的均值—>减均值->求协方差矩阵
        meanVals = mean(data, axis=0)
        meanRemoved = data - meanVals
        # 2.求协方差矩阵
        covMat = cov(meanRemoved, rowvar=0)
        # 3.计算协方差矩阵的特征值和特征向量
        eigVals, eigVects = linalg.eig(mat(covMat))
        # 4.将特征值从大到小排序
        eigValInd = argsort(eigVals)  # 对特征值进行排序，默认从小到大
        eigValInd = eigValInd[:-(k + 1):-1]  # 逆序取得特征值最大的元素
        # 5.保留前K个特征值对应的特征向量
        redEigVects = eigVects[:, eigValInd]  # 用特征向量构成矩阵
        # 6.将数据转换到K个特征向量构建的空间
        lowDDataMat = meanRemoved * redEigVects
        # 还原原始数据
        reconMat = (lowDDataMat * redEigVects.T) + meanVals
        return lowDDataMat, reconMat


if __name__ == '__main__':
    p = PCA()
    data = p.getData('data.txt')
    res = p.pca(data)
    print(res[0])