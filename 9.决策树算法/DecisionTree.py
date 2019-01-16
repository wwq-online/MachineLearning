# -*-coding:utf-8-*-
import numpy as np
import math
from Node import Node

"""
Author: wuweiqun
机器学习经典算法：决策树算法
参考李航《统计学习方法》， Chapter5 决策树
算法步骤：
    1. 若训练数据集D所有样本同属于一类C_k, 则决策树T为单节点树，并将类C_k作为该节点的类标记
    2. 若特征集A为空，则决策树T为单节点树, 并将训练数据集D中样本数最大的类C_k作为该节点的类标记
    3. 否则，计算特征集A中对训练数据集D的信息增益，选择为信息增益最大的特征A_g
    4. 信息增益最大特征A_g的信息增益小于阈值eta，则决策树T为单节点树，并将训练数据集D中样本数最大的类C_k作为该节点的类标记
    5. 否则，将信息增益最大特征A_g的每一个可能值a_i，按照特征集中A_g=a_i将训练数据集划分为若干个非空子集D_i,将非空子集
    6. 否则，将信息增益最大特征A_g的每一个可能值a_i，按照特征集中A_g=a_i将训练数据集划分为若干个非空子集D_i,将非空子集
    D_i中的样本数最大的类作为标记，构建子节点，由节点及其子节点构成树
"""

class DecisionTree():
    def __init__(self):
        self.trainData = self.getData()
        self.epsilon = 0.1

    def getData(self, dataPath='./data.txt'):
        """
        获取训练样本
        :param dataPath: 训练样本路径
        :return: 为训练数据T(X,Y),维度为(16, 5)
        """
        return np.loadtxt(dataPath, dtype=str, encoding='utf-8')

    def getEntropy(self, data):
        """
        计算信息熵
        :param data:当前数据集
        :return:
        """
        numSamples = data.shape[0]
        # 计算当前数据集中的每个类别数目
        numLableCounts = {}
        for i in range(numSamples):
            label = data[i][-1]
            if label not in numLableCounts:
                numLableCounts[label] = 0
            numLableCounts[label] += 1
        # 信息熵计算公式-Ep*log(p)
        entropy = 0.0
        for i in numLableCounts.values():
            prob = i / numSamples
            entropy += prob * math.log(prob, 2)
        return -entropy

    def getCondEntropy(self, data, featureIndex):
        """
        计算条件熵
        :param data: 数据集
        :param featureIndex: 特征索引
        :return: 条件熵值
        """
        numSamples = data.shape[0]
        featureSets = {}
        for i in range(numSamples):
            feature = data[i][featureIndex]
            if feature not in featureSets:
                featureSets[feature] = []

            featureSets[feature].append(data[i])
        condEntropy = 0.0
        for subData in featureSets.values():
            numSubSamples = len(subData)
            prob = numSubSamples / numSamples
            entropy = self.getEntropy(np.asarray(subData))
            condEntropy += prob * entropy
        return condEntropy

    def getInfoGain(self, entropy, condEbtropy):
        return entropy - condEbtropy

    def getFeatureSelected(self, data):
        """
        根据信息增益，选择信息增益最大的特征
        :param data: 数据集
        :return: 特征索引与信息增益值
        """
        numFeatures = data.shape[1] - 1
        entropy = self.getEntropy(data)
        # 计算每个特征的信息增益
        bestInfoGain = 0.0
        bestFeatureIndex = 0
        for featureIndex in range(numFeatures):
            infoGain = self.getInfoGain(entropy, self.getCondEntropy(data, featureIndex))
            if bestInfoGain < infoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = featureIndex
        selecedFeature = [bestFeatureIndex, bestInfoGain]
        return selecedFeature

    def train(self, data):
        features = data[0, :]
        yLabel = data[1:, -1]
        xData = data[1:, :-1]
        xAndY = data[1:, :]
        yLabelSet = {}

        for i in yLabel:
            if i not in yLabelSet.keys():
                yLabelSet[i] = 1
            else:
                yLabelSet[i] += 1

        # (1) 若训练数据集D所有样本同属于一类C_k, 则决策树T为单节点树，并将类C_k作为该节点的类标记
        if len(yLabelSet) == 1:
            print('所有样本同属于一类！')
            return Node(root=True, label=yLabel[0])

        # (2) 若特征集A为空，则决策树T为单节点树, 并将训练数据集D中样本数最大的类C_k作为该节点的类标记
        if len(features) == 0:
            print('特征集为空，改节点为叶子节点！')
            maxNumClass = max(yLabelSet.values())
            for i in yLabelSet:
                if yLabelSet[i] == maxNumClass:
                    return Node(root=True, label=i)

        # (3) 否则，计算特征集A中对训练数据集D的信息增益，选择为信息增益最大的特征A_g
        bestFeatureIndex, bestInfoGain = self.getFeatureSelected(xAndY)
        bestFeature = features[bestFeatureIndex]

        # (4) 信息增益最大特征A_g的信息增益小于阈值eta，则决策树T为单节点树，并将训练数据集D中样本数最大的类C_k作为该节点的类标记
        if bestInfoGain < self.epsilon:
            print('最大信息增益小于阈值！')
            maxNumClass = max(yLabelSet.values())
            for i in yLabelSet:
                if yLabelSet[i] == maxNumClass:
                    return Node(root=True, label=i)
        # (5) 否则，将信息增益最大特征A_g的每一个可能值a_i，按照特征集中A_g=a_i将训练数据集划分为若干个非空子集D_i,将非空子集
        # D_i中的样本数最大的类作为标记，构建子节点，由节点及其子节点构成树
        nodeTree = Node(root=False, featureName=bestFeature, feature=bestFeatureIndex)
        # 划分数据集子集
        bestFeatureValues = list(set(xAndY[:, bestFeatureIndex]))
        for value in bestFeatureValues:
            subfeatures = np.delete(features.reshape((1, -1)), bestFeatureIndex, axis=1)
            subData = np.delete(data[data[:, bestFeatureIndex] == value], bestFeatureIndex, axis=1)
            subData = np.concatenate([subfeatures, subData], axis=0)

            # 递归生成树
            subTree = self.train(subData)
            nodeTree.addNode(value, subTree)
        return nodeTree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)


if __name__ == '__main__':
    decisionTree = DecisionTree()
    # print('Training Data', decisionTree.getData())
    # print('Information Gain', decisionTree.getEntropy(decisionTree.trainData))
    # print('Conditional Entropy', decisionTree.getCondEntropy(decisionTree.trainData, 0))
    # print('Information Gain', decisionTree.getCondEntropy(decisionTree.trainData, 4))
    # print('Decision Train', decisionTree.getCondEntropy(decisionTree.trainData, 4))
    decisionTree.fit(decisionTree.trainData)
    print(decisionTree._tree)