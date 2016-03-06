from numpy import unique, array, zeros, argmin, delete, arange
from collections import Counter
from utils.print import Print
import math

__author__ = 'Simon & Oskar'

class DecisionTree:
    def __init__(self, criterion = 'gini', max_feature = None, max_depth = None, min_samples_leaf = 1, laplace = 0):
        self.criterion = criterion # gini or entropy
        self.max_feature = max_feature # random subspace size
        self.max_depth = max_depth  # maximum leaf depth
        self.min_samples_leaf = min_samples_leaf # minimum leaf size
        self.laplace = laplace # laplace correction for probability estimation

        self.value = None
        self.left_child = None
        self.right_child = None

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
        left_split = x[:, self.value[0]] <= self.value[1]
        right_split = x[:, self.value[0]] > self.value[1]
        l = x[left_split]
        r = x[right_split]
        self.left_child = self.fit(l, y[left_split], cur_max_depth, self.min_samples_leaf, self.max_features)
        self.right_child = self.fit(r, y[right_split], cur_max_depth, self.min_samples_leaf, self.max_features)

        return self

    def predict(self, x): # classify objects
        result = list()
        for row in x:
            current_node = self.tree
            while True:
                if type(current_node) is type(self):
                    if row[current_node.value[0]] <= current_node.value[1]:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
                else:
                    most_common_element = Counter(current_node.tolist()).most_common(1)
                    result.append(most_common_element[0][0])
                    break
        return array(result)

    def predict_proba(self, x): # class probability estimation
        result = list()
        for row in x:
            current_node = self.tree
            while(True):
                if type(current_node) is type(self):
                    if row[current_node.value[0]] <= current_node.value[1]:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
                else:
                    r = zeros(self.n_classes.shape).astype('float')
                    for class_element in current_node:
                        for i in range(len(self.n_classes)):
                            if class_element == self.n_classes[i]:
                                r[i] += 1.0 / float(len(current_node))
                    result.append(r)
                    break
        return array(result)

    def split(self, features, classes):
        s = list()
        for i in range(features.shape[1]):
            feature_data = features[:, i]
            unique_data = unique(feature_data)
            unique_data = delete(unique_data, arange(0, unique_data.size, 1.2))
            for splitIndex in unique_data:
                leftNodes = feature_data[feature_data <= splitIndex]
                rightNodes = feature_data[feature_data > splitIndex]
                if leftNodes.size == 0 or rightNodes.size == 0: continue
                s.append([i, splitIndex, self.ginisplit(leftNodes, rightNodes, (classes[:leftNodes.size], classes[leftNodes.size:]))])
                if len(s) >= self.max_features:
                    break
        return s

    def gini(self, x, y):
       unique_value = unique(y)
       result = 0
       for value in unique_value:
           result += math.pow(self.probability((y[y == value]).shape[0], (y[y != value]).shape[0]), 2)
       return 1.0 - result

    def probability(self, a, b):
       if a+b == 0: return 0
       return float(a)/float((a+b))

    def ginisplit(self, leftNodes, rightNodes, y):
       precords = float(leftNodes.size + rightNodes.size)
       return (float(leftNodes.size)/precords) * self.gini(leftNodes, y[0]) + (float(rightNodes.size) / precords) * self.gini(rightNodes, y[1])

    def laplace(self, x):
       result = []
       unique_classes = unique(x)
       for c in unique_classes:
           t = x[x == c]
           result.append(float(t.shape[0] + 1) / float(x.shape[0] + unique_classes.shape[0]))
       return result

    def print(self): # visualize tree (console)
        depth = 0
        Print.tree(self, depth)
