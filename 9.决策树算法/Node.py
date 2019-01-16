
class Node:
    def __init__(self, root=True, label=None, featureName=None, feature=None):
        self.root = root
        self.label = label
        self.featureName = featureName
        self.feature = feature
        self.tree = {}
        self.result = {'label': self.label, 'feature': self.feature, 'featureName': self.featureName, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def addNode(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)