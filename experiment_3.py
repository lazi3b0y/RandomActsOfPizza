from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_current_data_set
from utils.parse import parse_csv
from scipy.stats import binom

import sklearn
import numpy
import warnings

__author__ = 'Simon & Oskar'


def experiment_3():
    data_set = "resources/raop.csv"

    classifiers = {
        "custom_decision_tree": DecisionTree(min_samples_leaf=10,
                                             max_depth=50),
        "sklearn_decision_tree": DecisionTreeClassifier(min_samples_leaf=10,
                                                        max_depth=50),
        "custom_random_forest": RandomForest(n_estimators=50,
                                             max_depth=40),
        "sklearn_random_forest": RandomForestClassifier(n_estimators=50,
                                                        max_depth=40),
        "sklearn_neighbors": KNeighborsClassifier(n_neighbors=20,
                                                  leaf_size=20,
                                                  algorithm='kd_tree'),
    }

    classes, features = parse_csv(data_set)

    # 'Cross Validation' parameters
    n_folds = 10
    n_elements = features.shape[0]

    # The 10x10 Fold Cross Validation.
    cross_val = sklearn.cross_validation.KFold(n=n_elements,
                                               n_folds=n_folds)

    print_current_data_set(data_set)
    corr_predictions = {}

    for key, classifier in classifiers.items():
        corr_predictions[key] = 0
        for train, test in cross_val:
            train_features = features[train]
            train_classes = classes[train]

            test_features = features[test]
            test_classes = classes[test]

            classifier.fit(X=train_features,
                           y=train_classes.ravel())

            prediction = classifier.predict(test_features)

            test_classes = test_classes.astype(numpy.float).ravel()
            prediction = prediction.astype(numpy.float)

            temp = 0
            for i in range(len(test_classes)):
                if test_classes[i] == prediction[i]:
                    temp += 1

            corr_predictions[key] += temp

    for key, value in corr_predictions.items():
        for key_2, value_2 in corr_predictions.items():
            if key_2 != key:
                print("{} vs {}: {}". format(key, key_2, mcnemar_midp(value, value_2)))


def mcnemar_midp(b, c):
    # Found at https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp

if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    experiment_3()
