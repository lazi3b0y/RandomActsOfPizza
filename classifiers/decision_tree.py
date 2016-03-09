from pandas import read_csv
from numpy import *
from collections import Counter
from sklearn import tree
import math as math
from scipy.stats import laplace


class BinaryTreeClassifier:
    def __init__(self, max_features=None, max_depth=None, min_samples_leaf=1, lap=0):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lap = lap

    def fit(self, x, y):
        self.n_classes = unique(y)
        self.tree = DecisionTree().buildTree(x,y, self.max_depth, self.min_samples_leaf, self.max_features)

    def predict(self, x):
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
                    #cn = [c for c in currentNode]
                    r = Counter(currentNode.tolist()).most_common(1)
                    result.append(r[0][0])
                    break
        return array(result)

    def predict_proba(self, x):
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

    def printTree(self):
        tree = self.tree
        self.pTree(tree, 0)

    def pTree(self, node, depth):
        if type(node) is type(DecisionTree()):
            v = "+--"
            for i in range(depth):
                v = "|  " + v
            v += " " + array_str(node.value)
            print(v)
            self.pTree(node.leftChild, depth + 1)
            self.pTree(node.rightChild, depth + 1)
        else:
            v = "+-"
            for i in range(depth):
                v = "|  " + v
            v += "["
            #for n in node:
            v += str(Counter(node).most_common(2)) + ", "
            v = v[:-2] + "]"
            print(v)

class DecisionTree:

    def __init__(self):
        self.value = None
        self.leftChild = None
        self.rightChild = None

    def buildTree(self, dataSet, classLabel, max_depth, min_samples_leaf, max_features):
        new_max_depth = max_depth
        if max_features is None:
            self.max_features = dataSet.size
        elif max_features > dataSet.size:
            self.max_features = dataSet.size
        else:
            self.max_features = max_features

        if(max_depth is not None and max_depth <= 0):
            return classLabel

        if(max_depth is not None):
            new_max_depth = max_depth - 1

        if(len(dataSet) <= min_samples_leaf):
            return classLabel

        if unique(classLabel).size == 1:
            return classLabel

        splits = array(self.split(dataSet, classLabel))

        if(splits.size <= 0):
            return classLabel

        i = argmin(splits[:, 2])
        self.value = splits[i]
        leftSplit = dataSet[:, self.value[0]] <= self.value[1]
        rightSplit = dataSet[:, self.value[0]] > self.value[1]
        l = dataSet[leftSplit]
        r = dataSet[rightSplit]
        self.leftChild = DecisionTree().buildTree(l, classLabel[leftSplit], new_max_depth, min_samples_leaf, max_features)
        self.rightChild = DecisionTree().buildTree(r, classLabel[rightSplit], new_max_depth, min_samples_leaf, max_features)

        return self

    def split(self, features, classes):
        s = list()
        for i in range(features.shape[1]):
            f = features[:, i]
            u = unique(f)
            #if u.size > 10:
            u = delete(u, arange(0, u.size, 1.2))
            for splitI in u:
                l = f[f <= splitI]
                r = f[f > splitI]
                if l.size == 0 or r.size == 0: continue
                s.append([i, splitI, self.ginisplit(l, r, (classes[:l.size], classes[l.size:]))])
                if len(s) >= self.max_features:
                    break
        return s


    def gini(self, x, y):
        u = unique(y)
        result = 0
        for value in u:
            result += math.pow(self.probability((y[y == value]).shape[0], (y[y != value]).shape[0]), 2)
        return 1.0 - result


    def probability(self, a, b):
        if a+b == 0: return 0
        return float(a)/float((a+b))

    def ginisplit(self, n1, n2, y):
        precords = float(n1.size + n2.size)
        return (float(n1.size)/precords) * self.gini(n1, y[0]) + (float(n2.size) / precords) * self.gini(n2, y[1])

    def laplace(self, x):
        result = []
        classes = unique(x)
        for c in classes:
            t = x[x == c]
            result.append(float(t.shape[0] + 1) / float(x.shape[0] + classes.shape[0]))
        return result