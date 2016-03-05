from numpy import unique, array, zeros, array_str
from collections import Counter

__author__ = 'Simon & Oskar'

class DecisionTree:
    def __init__(self, criterion = 'gini', max_feature = None, max_depth = None, min_samples_leaf = 1, laplace = 0):
        self.criterion = criterion # gini or entropy
        self.max_feature = max_feature # random subspace size
        self.max_depth = max_depth  # maximum leaf depth
        self.min_samples_leaf = min_samples_leaf # minimum leaf size
        self.laplace = laplace # laplace correction for probability estimation

    def fit(self, x, y): # train model
        self.n_classes = unique(y)

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
                    #cn = [c for c in currentNode]
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

    def print(self): # visualize tree (console)
        tree = self.tree
        depth = 0
        if type(tree) is type(DecisionTree()):
            v = "+--"
            for i in range(depth):
                v = "|  " + v
            v += " " + array_str(tree.value)
            print(v)
            self.pTree(tree.leftChild, depth + 1)
            self.pTree(tree.rightChild, depth + 1)
        else:
            v = "+-"
            for i in range(depth):
                v = "|  " + v
            v += "["
            #for n in node:
            v += str(Counter(tree).most_common(2)) + ", "
            v = v[:-2] + "]"
            print(v)