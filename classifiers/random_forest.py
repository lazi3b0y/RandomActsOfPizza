import numpy
from random import randrange
from classifiers.decision_tree import DecisionTree
from collections import Counter

__author__ = 'Simon & Oskar'


# TODO: Rename variables, restructure code(?), comment code
class RandomForest:
    def __init__(self, max_depth = None, min_samples_leaf = 1, n_estimators = 10, sample_size = 200, max_features = None):
        self.criterion = "gini"
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.laplace = 0
        self.n_estimators = n_estimators
        self.bagging = 0
        self.sample_size = sample_size

        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = numpy.unique(y)

        if self.sample_size is None:
            self.sample_size = X.shape[0] / self.n_estimators

        random_samples = numpy.zeros((self.n_estimators, self.sample_size, X.shape[1]), numpy.float)
        random_samples_class = numpy.zeros((self.n_estimators, self.sample_size))

        for i in range(self.n_estimators):
            temp = numpy.zeros((self.sample_size, X.shape[1]), numpy.float)
            temp_class = numpy.zeros((self.sample_size), numpy.float)
            for j in range(self.sample_size):
                r = randrange(0,X.shape[0])
                temp[j] = X[r]
                temp_class[j] = y[r]
            random_samples[i] = temp
            random_samples_class[i] = temp_class

        self.trees = list()
        for i in range(self.n_estimators):
            tree = DecisionTree(max_features = self.max_features,
                                max_depth = self.max_depth,
                                min_samples_leaf = self.min_samples_leaf,
                                laplace = self.laplace)
            tree.fit(random_samples[i], random_samples_class[i])
            self.trees.append(tree)

    def predict(self, X):
        final_result = numpy.zeros((X.shape[0]))

        for row in range(X.shape[0]):
            result = numpy.zeros((self.n_estimators, 1))
            for i in range(self.n_estimators):
                result[i] = self.trees[i].predict(numpy.array([X[row]]))
            cn = [r[0] for r in result]
            c = Counter(cn).most_common(numpy.unique(cn).size)
            final_result[row] = self.pick_result(c)

        return final_result

    def predict_proba(self, X):
        final_result = numpy.zeros((X.shape[0], len(self.n_classes)), numpy.float)

        for row in range(X.shape[0]):
            result = numpy.zeros((self.n_estimators, len(self.n_classes)), numpy.float)
            for i in range(self.n_estimators):
                result[i] = self.trees[i].predict_proba(numpy.array([X[row]]))
            result.astype(numpy.float)
            for r in result:
                for i in range(final_result.shape[1]):
                    final_result[row, i] += r[i].astype(numpy.float) / float(self.n_estimators)

        return final_result

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
