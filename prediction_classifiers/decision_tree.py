from numpy import unique, array, zeros, array_str, argmin
from collections import Counter
from utils.print import Print

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
                if type(currentNode) is type(self):
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
                if type(currentNode) is type(self):
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

    #RÖVVVVVVVVVVVVV
    #def gini(self, x, y):
    #    u = unique(y)
    #    result = 0
    #    for value in u:
    #        result += math.pow(self.probability((y[y == value]).shape[0], (y[y != value]).shape[0]), 2)
    #    return 1.0 - result


    #def probability(self, a, b):
    #    if a+b == 0: return 0
    #    return float(a)/float((a+b))

    #def ginisplit(self, n1, n2, y):
    #    precords = float(n1.size + n2.size)
    #    return (float(n1.size)/precords) * self.gini(n1, y[0]) + (float(n2.size) / precords) * self.gini(n2, y[1])

    #def laplace(self, x):
    #    result = []
    #    classes = unique(x)
    #    for c in classes:
    #        t = x[x == c]
    #        result.append(float(t.shape[0] + 1) / float(x.shape[0] + classes.shape[0]))
    #    return result

    def print(self): # visualize tree (console)
        depth = 0
        Print.tree(self, depth)
