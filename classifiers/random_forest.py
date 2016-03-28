from classifiers.decision_tree import DecisionTree
from collections import Counter
from utils.utility import pick_result
from random import randrange

import numpy

__author__ = 'Simon & Oskar'


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
        self.decision_trees = list()

    def fit(self, X, y):
        if self.n_classes is None:
            self.n_classes = numpy.unique(y)

        if self.sample_size is None:
            self.sample_size = X.shape[0] / self.n_estimators

        rand_features = numpy.zeros(shape = (self.n_estimators, self.sample_size, X.shape[1]),
                                     dtype = numpy.float)

        rand_classes = numpy.zeros((self.n_estimators, self.sample_size))

        for i in range(self.n_estimators):
            temp = numpy.zeros(shape = (self.sample_size, X.shape[1]),
                               dtype = numpy.float)

            temp_class = numpy.zeros(shape = self.sample_size,
                                     dtype = numpy.float)

            for j in range(self.sample_size):
                r = randrange(0, X.shape[0])
                temp[j] = X[r]
                temp_class[j] = y[r]

            rand_features[i] = temp
            rand_classes[i] = temp_class

        for i in range(self.n_estimators):
            decision_tree = DecisionTree(max_features = self.max_features,
                                         max_depth = self.max_depth,
                                         min_samples_leaf = self.min_samples_leaf,
                                         laplace = self.laplace)

            decision_tree.fit(X = rand_features[i],
                              y = rand_classes[i])

            self.decision_trees.append(decision_tree)

    def predict(self, X):
        final_result = numpy.zeros((X.shape[0]))

        for row in range(X.shape[0]):
            result = numpy.zeros((self.n_estimators, 1))

            for i in range(self.n_estimators):
                result[i] = self.decision_trees[i].predict(numpy.array([X[row]]))

            result_flattened = result.ravel()
            c = Counter(result_flattened).most_common(numpy.unique(result_flattened).size)

            final_result[row] = pick_result(c)

        return final_result

    def predict_proba(self, X):
        final_result = numpy.zeros((X.shape[0], len(self.n_classes)), numpy.float)

        for row in range(X.shape[0]):
            result = numpy.zeros((self.n_estimators, len(self.n_classes)), numpy.float)
            for i in range(self.n_estimators):
                result[i] = self.decision_trees[i].predict_proba(numpy.array([X[row]]))
            result.astype(numpy.float)
            for r in result:
                for i in range(final_result.shape[1]):
                    final_result[row, i] += r[i].astype(numpy.float) / float(self.n_estimators)

        return final_result
