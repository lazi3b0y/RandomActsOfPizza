from numpy import unique, array, zeros, array_str, argmin
from collections import Counter

__author__ = 'Simon & Oskar'

class DecisionTree:
    def __init__(self, criterion = 'gini', max_feature = None, max_depth = None, min_samples_leaf = 1, laplace = 0):
        self.criterion = criterion # gini or entropy
        self.max_feature = max_feature # random subspace size
        self.max_depth = max_depth  # maximum leaf depth
        self.min_samples_leaf = min_samples_leaf # minimum leaf size
        self.laplace = laplace # laplace correction for probability estimation

        self.value = None
        self.leftChild = None
        self.rightChild = None

    def fit(self, x, y): # train model
        self.n_classes = unique(y)

        cur_max_depth = self.max_depth
        if self.max_features is None:
            self.max_features = x.size
        elif self.max_features > x.size:
            self.max_features = x.size
        else:
            self.max_features = self.max_features

        if self.max_depth is not None and self.max_depth <= 0:
            return y

        if self.max_depth is not None:
            cur_max_depth = self.max_depth - 1

        if len(x) <= self.min_samples_leaf:
            return y

        if unique(y).size == 1:
            return y

        splits = array(self.split(x, y))

        if splits.size <= 0:
            return y

        i = argmin(splits[:, 2])
        self.value = splits[i]
        leftSplit = x[:, self.value[0]] <= self.value[1]
        rightSplit = x[:, self.value[0]] > self.value[1]
        l = x[leftSplit]
        r = x[rightSplit]
        self.leftChild = self.fit(l, y[leftSplit], cur_max_depth, self.min_samples_leaf, self.max_features)
        self.rightChild = self.fit(r, y[rightSplit], cur_max_depth, self.min_samples_leaf, self.max_features)

        return self

    def predict(self, x): # classify objects
        result = list()
        for row in x:
            currentNode = self.tree
            while True:
                if type(currentNode) is type(DecisionTree()):
                    if row[currentNode.value[0]] <= currentNode.value[1]:
                        currentNode = currentNode.leftChild
                    else:
                        currentNode = currentNode.rightChild
                else:
                    r = Counter(currentNode.tolist()).most_common(1)
                    result.append(r[0][0])
                    break
        return array(result)

    def predict_proba(self, x): # class probability estimation
        result = list()
        for row in x:
            currentNode = self.tree
            while(True):
                if type(currentNode) is type(DecisionTree()):
                    if row[currentNode.value[0]] <= currentNode.value[1]:
                        currentNode = currentNode.leftChild
                    else:
                        currentNode = currentNode.rightChild
                else:
                    r = zeros(self.n_classes.shape).astype('float')
                    for c in currentNode:
                        for i in range(len(self.n_classes)):
                            if c == self.n_classes[i]:
                                r[i] += 1.0 / float(len(currentNode))
                    result.append(r)
                    break
        return array(result)

    def print(self, tree, depth = 0): # visualize tree (console)
        if type(tree) is type(DecisionTree()):
            v = "+--"
            for i in range(depth):
                v = "|  " + v
            v += " " + array_str(tree.value)
            print(v)
            self.print(tree.leftChild, depth + 1)
            self.print(tree.rightChild, depth + 1)
        else:
            v = "+-"
            for i in range(depth):
                v = "|  " + v
            v += "["
            v += str(Counter(tree).most_common(2)) + ", "
            v = v[:-2] + "]"
            print(v)