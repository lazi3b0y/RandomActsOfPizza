from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_current_data_set, print_mcnemar, print_accuracies
from utils.parse import parse_csv

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
    accuracies = {}

    for key, classifier in classifiers.items():
        result = list()
        corr_predictions[key] = 0
        accuracies[key] = 0.0
        avg_accuracy = 0.0

        for train, test in cross_val:
            train_features = features[train]
            train_classes = classes[train]

            test_features = features[test]
            test_classes = classes[test]

            classifier.fit(X=train_features,
                           y=train_classes.ravel())

            prediction = classifier.predict(test_features)

            test_classes = test_classes.astype(numpy.float)
            prediction = prediction.astype(numpy.float)

            result.append([test_classes, prediction])

            test_classes = test_classes.ravel()

            temp = 0
            for i in range(len(test_classes)):
                if test_classes[i] == prediction[i]:
                    temp += 1

            corr_predictions[key] += temp

        for row in range(len(result)):
            avg_accuracy += sklearn.metrics.accuracy_score(y_true=result[row][0],
                                                           y_pred=result[row][1])

        avg_accuracy /= float(len(result))

        accuracies[key] += avg_accuracy

    print_accuracies(accuracies)

    print_mcnemar(corr_predictions)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    experiment_3()
