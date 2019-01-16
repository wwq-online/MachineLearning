# -*-coding:utf-8-*-
import numpy as np
"""
Author: wuweiqun
机器学习经典算法：朴素贝叶斯分类器算法
参考李航《统计学习方法》， Chapter4 朴素贝叶斯法
算法步骤：
    1. 计算先验概率
    2. 计算条件概率
    3. 根据特征独立假设，对给定的测试样本计算其后验概率P(Y=y|X=x)
    4. 将后验概率最大的类作为输出分类
"""


class NaiveBayes():
    def __init__(self, lamda=0.0):

        """
        初始化
        trainingData为训练数据T(X,Y),维度为(15, 3), 15表示样本数目; 3表示2个特征x1,x2+ 1个标签，数据类型为字符串numpy.ndarray
        x1Set为特征x1的取值集合，数据类型为List
        x2Set为特征x2的取值集合，数据类型为List
        ySet为标签的取值集合，数据类型为List
        priorProb为根据样本计算得到的先验概率
        condProb为根据样本计算得到的条件概率
        :param lamda: lamda为0时，为极大似然估计; lamda>0时，贝叶斯估计，特殊值lamda为1时，表示拉普拉斯平滑

        """
        self.trainData = self.getData()
        self.x1Set = list(set(self.trainData[:, 0]))
        self.x2Set = list(set(self.trainData[:, 1]))
        self.ySet = list(set(self.trainData[:, 2]))
        self.priorProb = self.getPriorProb(lamda)
        self.condProb = self.getCondProb(lamda)


    def getData(self, dataPath='./data.txt'):
        """
        获取训练样本
        :param dataPath: 训练样本路径
        :return: 为训练数据T(X,Y),维度为(15, 3), 15表示样本数目; 3表示2个特征x1,x2+ 1个标签，数据类型为字符串numpy.ndarray
        """
        return np.loadtxt(dataPath, dtype=str, encoding='utf-8')[1:]

    def getPriorProb(self, lamda):
        """
        计算先验概率
        :param lamda: lamda为0时，为极大似然估计; lamda>0时，贝叶斯估计，特殊值lamda为1时，表示拉普拉斯平滑
        :return: 先验概率字典，key为类别，value为对应类别先验概率值
        """
        # 样本标签数据
        y = self.trainData[:, 2]
        # 样本数目
        numSanmples = y.shape[0]
        # 类别1样本数目与类别2样本数目
        numClass1 = y[y == str(self.ySet[0])].shape[0]
        numClass2 = y[y == str(self.ySet[1])].shape[0]
        # K表示类别数目
        K = len(self.ySet)

        return{self.ySet[0]: (numClass1 + lamda) / (numSanmples + K * lamda), self.ySet[1]: (numClass2 + lamda) / (numSanmples + K * lamda)}

    def getCondProb(self, lamda):
        """
        计算条件概率
        :param lamda: lamda为0时，为极大似然估计; lamda>0时，贝叶斯估计，特殊值lamda为1时，表示拉普拉斯平滑
        :return: 条件概率字典，key为"特征|类别",value为对应条件概率值
        """
        # 条件概率字典
        conProbDict = {}
        # 类别1样本数据与类别2样本数据
        samplesClass1 = self.trainData[self.trainData[:, 2] == str(self.ySet[0])]
        samplesClass2 = self.trainData[self.trainData[:, 2] == str(self.ySet[1])]
        # 类别1样本数目与类别2样本数目
        numClass1 = samplesClass1.shape[0]
        numClass2 = samplesClass2.shape[0]

        # 计算条件概率P(x1=x1_i|Y_j=y)
        for i in self.x1Set:
            pX1 = samplesClass1[samplesClass1[:, 0] == str(i)]
            nX1 = samplesClass2[samplesClass2[:, 0] == str(i)]
            # S表示特征集合数目(取值数目)
            S = len(self.x1Set)
            # P(X1=x1|Y=y)
            conProbDict[str(i) + '|' + str(self.ySet[0])] = (pX1.shape[0] + lamda) / (numClass1 + S * lamda)
            conProbDict[str(i) + '|' + str(self.ySet[1])] = (nX1.shape[0] + lamda) / (numClass2 + S * lamda)

        # 计算条件概率P(x2=x2_i|Y_j=y)
        for i in self.x2Set:
            # P(X2=x2|Y=y)
            pX2 = samplesClass1[samplesClass1[:, 1] == str(i)]
            nX2 = samplesClass2[samplesClass2[:, 1] == str(i)]
            # S表示特征集合数目(取值数目)
            S = len(self.x2Set)
            conProbDict[str(i) + '|' + str(self.ySet[0])] = (pX2.shape[0] + lamda) / (numClass1 + S * lamda)
            conProbDict[str(i) + '|' + str(self.ySet[1])] = (nX2.shape[0] + lamda) / (numClass2 + S * lamda)

        return conProbDict

    def predict(self, x):
        """
        二分类预测
        :param x: 输入单个样本数据，包含x1与x2两个特征的List
        :return: 预测类别，预测概率值(没有进行归一化)
        """
        prob1 = self.priorProb[str(self.ySet[0])] * self.condProb[str(x[0]) + '|' + str(self.ySet[0])] * self.condProb[str(x[1]) + '|' + str(self.ySet[0])]
        prob2 = self.priorProb[str(self.ySet[1])] * self.condProb[str(x[0]) + '|' + str(self.ySet[1])] * self.condProb[str(x[1]) + '|' + str(self.ySet[1])]
        result = [self.ySet[0], prob1] if (prob1 > prob2) else [self.ySet[1], prob2]
        className = result[0]
        score = result[1]
        return className, score


if __name__ == '__main__':
    # lamda为0时，为极大似然估计; lamda>0时，贝叶斯估计，特殊值lamda为1时，表示拉普拉斯平滑
    naiveBayes = NaiveBayes(lamda=0.0)
    # print('Prior Probability:', naiveBayes.getPriorProb(1))
    # print('Conditional Probability:', naiveBayes.getCondProb(1))
    print('Result od Prediction:', naiveBayes.predict(['2', 'S']))
    naiveBayes = NaiveBayes(lamda=1.0)
    print('Result od Prediction:', naiveBayes.predict(['2', 'S']))




