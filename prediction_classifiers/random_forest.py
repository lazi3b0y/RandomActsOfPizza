from numpy import unique

__author__ = 'Simon & Oskar'

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

    def fit(self, x, y): # train model
        self.n_classes = unique(y)
