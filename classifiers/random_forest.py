import numpy
from random import randrange
from classifiers.decision_tree import DecisionTree
from collections import Counter

__author__ = 'Simon & Oskar'

# TODO: Rename variables, restructure code(?), comment code
class RandomForest:
    def __init__(self, max_depth=None, min_samples_leaf=1, n_estimators=10, sample_size=200):
        self.criterion = "gini"
        self.max_features = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.laplace = 0
        self.n_estimators = n_estimators
        self.bagging = 0
        self.sample_size = sample_size

    def fit(self, x, y):
        self.classType = y.dtype
        self.n_classes = numpy.unique(y)
        if self.sample_size == None:
            self.sample_size = x.shape[0] / self.n_estimators

        random_samples = numpy.zeros((self.n_estimators, self.sample_size, x.shape[1]), 'float64')
        random_samples_class = numpy.zeros((self.n_estimators, self.sample_size), self.classType)
        for i in range(self.n_estimators):
            temp = numpy.zeros((self.sample_size, x.shape[1]), 'float64')
            temp_class = numpy.zeros((self.sample_size), self.classType)
            for j in range(self.sample_size):
                r = randrange(0,x.shape[0])
                temp[j] = x[r]
                temp_class[j] = y[r]
            random_samples[i] = temp
            random_samples_class[i] = temp_class

        self.trees = list()
        for i in range(self.n_estimators):
            tree = DecisionTree(max_features = self.max_features, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, laplace = self.laplace)
            tree.fit(random_samples[i], random_samples_class[i])
            self.trees.append(tree)


    def predict(self, x):
        finalResult = numpy.zeros((x.shape[0]), self.classType)
        for row in range(x.shape[0]):
            result = numpy.zeros((self.n_estimators, 1))
            result = result.astype(self.classType)
            for i in range(self.n_estimators):
                result[i] = self.trees[i].predict(numpy.array([x[row]]))
            cn = [r[0] for r in result]
            c = Counter(cn).most_common(numpy.unique(cn).size)
            finalResult[row] = self.pick_result(c)

        return finalResult

    def predict_proba(self, x):
        finalResult = numpy.zeros((x.shape[0], len(self.n_classes)), 'float')
        for row in range(x.shape[0]):
            result = numpy.zeros((self.n_estimators, len(self.n_classes)), 'float')
            for i in range(self.n_estimators):
                result[i] = self.trees[i].predict_proba(numpy.array([x[row]]))
            result.astype('float')
            for r in result:
                for i in range(finalResult.shape[1]):
                    finalResult[row, i] += r[i].astype('float') / float(self.n_estimators)

        return finalResult

    @staticmethod
    def pick_result(values):
        if len(values) != 1:
            equals = list()
            equals.append(0)
            for v in range(1, len(values)):
                if values[0][1] == values[v][1]:
                    equals.append(v)
            return values[equals[randrange(0, len(equals))]][0]
        else:
            return values[0][0]