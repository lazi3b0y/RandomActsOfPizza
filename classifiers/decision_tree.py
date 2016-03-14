from numpy import unique, argmin, array, zeros, delete, arange
from collections import Counter
from utils.print import print_tree
import math


class DecisionTree:
    def __init__(self, max_features=None, max_depth=None, min_samples_leaf=1, laplace=0):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.laplace = laplace

        self.value = None
        self.left_child = None
        self.right_child = None
        self.n_classes = None

    # x = rows of post stats
    # y = set of classes, i.e. if a post recieved a pizza or not.
    def fit(self, x, y):
        if self.n_classes is None:
            self.n_classes = unique(y)

        if self.max_depth is not None and self.max_depth == 0:
            return y

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

        left_child_tree = DecisionTree(min_samples_leaf=self.min_samples_leaf,
                                       max_depth=self.max_depth - 1 if self.max_depth is not None else self.max_depth,
                                       max_features=self.max_features)

        right_child_tree = DecisionTree(min_samples_leaf=self.min_samples_leaf,
                                        max_depth=self.max_depth - 1 if self.max_depth is not None else self.max_depth,
                                        max_features=self.max_features)

        self.left_child = left_child_tree.fit(x[left_split],
                                              y[left_split])

        self.right_child = right_child_tree.fit(x[right_split],
                                                y[right_split])

        return self

    def predict(self, x):  # classify objects
        result = list()
        for row in x:
            current_node = self
            while True:
                if isinstance(current_node, DecisionTree):
                    if row[current_node.value[0]] <= current_node.value[1]:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
                else:
                    most_common_element = Counter(current_node.tolist()).most_common(1)
                    result.append(most_common_element[0][0])
                    break
        return array(result)

    def predict_proba(self, x):  # class probability estimation
        result = list()
        for row in x:
            current_node = self
            while True:
                if isinstance(current_node, DecisionTree):
                    if row[current_node.value[0]] <= current_node.value[1]:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
                elif current_node is None:
                    break
                else:
                    r = zeros(self.n_classes.shape).astype('float')
                    for class_element in current_node:
                        for i in range(len(self.n_classes)):
                            if class_element == self.n_classes[i]:
                                r[i] += 1.0 / float(len(current_node))
                    result.append(r)
                    break
        return array(result)

    def print(self):  # Visualize Tree (console)
        depth = 0
        print_tree(self, depth)

    def split(self, x, y):
        s = list()
        for i in range(x.shape[1]):
            feature_data = x[:, i]
            unique_data = unique(feature_data)
            unique_data = delete(unique_data, arange(unique_data.size, 1.2))
            for splitIndex in unique_data:
                left_nodes = feature_data[feature_data <= splitIndex]
                right_nodes = feature_data[feature_data > splitIndex]
                if left_nodes.size == 0 or right_nodes.size == 0:
                    continue
                s.append(
                    [i, splitIndex,
                     self.ginisplit(left_nodes, right_nodes, (y[:left_nodes.size], y[left_nodes.size:]))])
                if self.max_features is None or len(s) >= self.max_features:
                    break
        return s

    def gini(self, x, y):
        uniq_values = unique(y)
        result = 0
        for value in uniq_values:
            result += math.pow(self.probability((y[y == value]).shape[0], (y[y != value]).shape[0]), 2)
        return 1.0 - result

    @staticmethod
    def probability(a, b):
        if a + b == 0: return 0
        return float(a) / float((a + b))

    def ginisplit(self, left_nodes, right_nodes, y):
        precords = float(left_nodes.size + right_nodes.size)
        return (float(left_nodes.size) / precords) * self.gini(left_nodes, y[0]) + (float(
            right_nodes.size) / precords) * self.gini(right_nodes, y[1])

    @staticmethod
    def laplace(x):
        result = []
        unique_classes = unique(x)
        for c in unique_classes:
            t = x[x == c]
            result.append(float(t.shape[0] + 1) / float(x.shape[0] + unique_classes.shape[0]))
        return result
