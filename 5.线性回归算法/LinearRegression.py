# -*-coding:utf-8-*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
"""
Author: wuweiqun
机器学习经典算法：线性回归算法
算法步骤：
    1. 
    2. 
    
"""
class LinearRegression():
    def __init__(self):
        self.trainData = self.getData()
        self.epsilon = 0.1

    def getData(self, dataPath='./data.txt'):
        """
        获取训练样本
        :param dataPath: 训练样本路径
        :return: 为训练数据T(X,Y),维度为(16, 5)
        """
        return np.loadtxt(dataPath, dtype=float, encoding='utf-8-sig', delimiter=',')

    def standRegres(self, xArr, yArr):
        """
        最小二乘法求权重w[0], w[1]
        :param xArr:
        :param yArr:
        :return:
        """
        xMat = mat(xArr)
        yMat = mat(yArr).T
        xTx = xMat.T * xMat
        ws = xTx.I * (xMat.T * yMat)
        return ws

if __name__ == '__main__':
    lr = LinearRegression()
    data = lr.getData('data.txt')
    x = data[:, 0:2]
    y = data[:, -1]
    w = lr.standRegres(x, y)
    print(w)