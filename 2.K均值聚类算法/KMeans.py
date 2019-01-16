# -*-coding:utf-8-*-
import numpy as np
import random

"""
Author: wuweiqun
机器学习经典算法：K均值聚类算法
参考赵志勇《Python机器学习算法》， Chapter10 K-Means
算法步骤：
    1. 随机初始化聚类中心
    2. 重复进行计算，直到聚类中心不再改变(体现在每个样本的分类不再变化)
        2.1 计算每个训练样本与当前聚类中心距离，并将与之最近的聚类中心作为该样本类别
        2.2 计算划分到每个类别的所有样本特征的均值，并将该均值作为每个类新的聚类中心
    3. 输出聚类中心
"""


class KMeans():
    def __init__(self, k=4):
        self.trainData = self.getData()
        self.k = k
        self.clusterCenters = self.getInitClusterCenters()

    def getData(self, dataPath='./data.txt'):
        """
        获取训练样本
        :param dataPath: 训练样本路径
        :return: 为训练数据T(X,Y),维度为(80, 2), 15表示样本数目; 2表示2个特征x1,x2, 数据类型为float numpy.ndarray
        """
        return np.loadtxt(dataPath, dtype=float, encoding='utf-8')

    def getInitClusterCenters(self):
        """
        随机初始化据类中心
        :return: 返回k个据类中心，维度为(k, numFeatures), k为聚类数目， nunFeatures为特征数目， 数据类型为float numpy.ndarray
        """
        numFeatures = self.trainData.shape[1]
        initClusterCenters = np.zeros((self.k, numFeatures), dtype=np.float)
        minFeatureValue = np.zeros(numFeatures)
        maxFeatureValue = np.zeros(numFeatures)
        # 随机初始化聚类中心

        # 计算特征最大值与最小值
        for i in range(numFeatures):
            minFeatureValue[i] = np.min(self.trainData[:, i])
            maxFeatureValue[i] = np.max(self.trainData[:, i])

        # 设置固定种子使得，固定随机选择据类中心
        random.seed(1.0)
        # 随机选择聚类中心
        for i in range(self.k):
            initClusterCenters[i, :] = np.asarray([random.uniform(minFeatureValue, maxFeatureValue)])

        return initClusterCenters

    def kmeansClustering(self):
        """

        :return:
        """

        numSamples, numFeatures = self.trainData.shape
        numClusterCenters = self.clusterCenters.shape[0]
        # 初始化所有样本的类别，其中1为类别索引
        samplesClasses = np.zeros((numSamples, 1), dtype=np.int)
        samplesClusterDists = np.zeros((numSamples, 1), dtype=np.float)

        notFinshed = True
        count = 0
        while notFinshed:
            notFinshed = False
            for i in range(numSamples):
                # 设置样本与聚类中心的初始最小值,类别索引
                minDist = np.Inf
                classIndex = 0

                # 计算样本与每个聚类中心的距离
                for j in range(numClusterCenters):
                    dist = self.getEuclideanDist(self.trainData[i], self.clusterCenters[j])
                    if dist < minDist:
                        minDist = dist
                        classIndex = j

                # 训练停止条件，所有聚类中心不再变化，体现在每个样本的分类不再变化
                if samplesClasses[i, 0] != classIndex:
                    notFinshed = True
                    # 将与之距离最小的聚类中心类设为该样本类别
                    samplesClasses[i, 0] = classIndex
                    samplesClusterDists[i, 0] = minDist

            # 重新计算聚类中心，计算方法为每个类样本的均值
            for i in range(numClusterCenters):
                sumFeatureValues = np.zeros(numFeatures, dtype=np.float)
                numSameClassSamples = 0
                for j in range(numSamples):
                    if samplesClasses[j, 0] == i:
                        sumFeatureValues += self.trainData[j, :]
                        numSameClassSamples += 1
                avgFeatureValues = sumFeatureValues / numSamples
                self.clusterCenters[i, :] = avgFeatureValues

            count += 1
            print('Epoch', count, self.clusterCenters)
        print('finished!')
        return samplesClasses, samplesClusterDists

    def getEuclideanDist(self, a, b):
        """
        :param a:
        :param b:
        :return: 欧式距离值
        """
        dist = np.sqrt(np.sum(np.square(a - b)))
        return dist

    def predict(self, x):
        """
        聚类预测
        :param x: 待测试样本
        :return: 预测聚类结果与距离
        """
        classIndex = 0
        minDist = np.inf
        for i in range(self.clusterCenters.shape[0]):
            dist = self.getEuclideanDist(self.clusterCenters[i, :], x)
            if dist < minDist:
                minDist = dist
                classIndex = i
        return classIndex, minDist


if __name__ == '__main__':
    kmeans = KMeans()
    print('Initial Cluster Centers', kmeans.clusterCenters)
    kmeans.kmeansClustering()
    print('Final Cluster Centers', kmeans.clusterCenters)
    classIndex, minDist = kmeans.predict(np.asarray([0.0, -0.6]))
    print('class:', classIndex, 'dist', minDist)
