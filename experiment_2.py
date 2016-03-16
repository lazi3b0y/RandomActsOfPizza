from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_clf_parameters
from utils.parse import parse_csv

import numpy
import sklearn

__author__ = "Simon & Oskar"


def experiment_2a():
    data_sets = [
        # "resources/multi_data_sets/iris.csv",
        "resources/multi_data_sets/glass.csv",
        "resources/multi_data_sets/vehicle.csv",
        "resources/multi_data_sets/segment.csv",
        "resources/binary_data_sets/balance-scale.csv",
        "resources/binary_data_sets/breast-cancer.csv",
        "resources/binary_data_sets/breast-w.csv",
        "resources/binary_data_sets/credit-a.csv",
        "resources/binary_data_sets/credit-g.csv",
        "resources/binary_data_sets/diabetes.csv",
        "resources/binary_data_sets/haberman.csv",
        "resources/binary_data_sets/heart-c.csv",
        "resources/binary_data_sets/heart-h.csv",
        "resources/binary_data_sets/heart-s.csv",
    ]

    # 'Decision Tree'/'Random Forest' parameters.
    max_depth = 30
    min_samples_leaf = 1
    n_estimators = 50
    max_features = None

    # 'KNeighbors' parameters
    n_neighbors = 5
    leaf_size = 30

    # Creation of the two Decision Tree Classifiers we'll be using during this experiment.
    # 'custom_decision_tree' is our own implementation of hunt's algorithm, while 'sklearn_decision_tree'
    # is scikit-learn's version of a Decision Tree Classifier.
    custom_decision_tree = DecisionTree(max_depth = max_depth,
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features)

    sklearn_decision_tree = DecisionTreeClassifier(max_depth = max_depth,
                                                   min_samples_leaf = min_samples_leaf,
                                                   max_features = max_features)

    # Creation of the two Random Forest Classifiers we'll be using during this experiment.
    # 'custom_random_forest' is our own implementation, while 'sklearn_random_forest'
    # is scikit-learn's version of a Decision Tree Classifier.
    custom_random_forest = RandomForest(n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_leaf = 30,
                                        max_features = max_features)

    sklearn_random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                                   max_depth = max_depth,
                                                   min_samples_leaf = 30,
                                                   max_features = max_features)

    sklearn_neighbors = KNeighborsClassifier(n_neighbors = n_neighbors,
                                             leaf_size = leaf_size)

    # A dictionary containing our classifiers that were initialized before.
    classifiers = {
        "custom_decision_tree": custom_decision_tree,
        "custom_random_forest": custom_random_forest,
        "sklearn_decision_tree": sklearn_decision_tree,
        "sklearn_random_forest": sklearn_random_forest,
        "sklearn_neighbors": sklearn_neighbors,
    }

    value_matrix_1 = [v for v in range(5, 51, 5)]
    value_matrix_2 = [v for v in range(30, 101, 10)]
    results = numpy.zeros((len(value_matrix_1), len(value_matrix_2)))

    # The actual experiment begins here.
    for data_set in data_sets:
        class_set, feature_set = parse_csv(data_set)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # The 10x10 Fold Cross Validation, divides the data set into training set and test set.
        # In our case we'll end up with a test set that 1/10 of the total data and a train set
        # thats 9/10 of the total data. Atleast as long as n_folds = 10.
        cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                                   n_folds = n_folds)

        print("Current data set:\t{}".format(data_set))

        for i in range(len(value_matrix_1)):
            for j in range(len(value_matrix_2)):
                # Prints the current parameters for our classifiers (Decision Trees/Random Forests/KNeighbors).
                print_clf_parameters(max_depth = value_matrix_1[i],
                                     min_samples_leaf = min_samples_leaf,
                                     n_estimators = value_matrix_2[j],
                                     n_neighbors = value_matrix_1[i],
                                     leaf_size = value_matrix_2[j])

                # classifiers["custom_random_forest"] = RandomForest(n_estimators = value_matrix_2[j],
                #                                                    max_depth = value_matrix_1[i])

                # classifiers["sklearn_random_forest"] = RandomForestClassifier(n_estimators = value_matrix_2[j],
                #                                                               max_depth = value_matrix_1[i])

                classifiers["sklearn_neighbors"] = KNeighborsClassifier(n_neighbors = value_matrix_1[i],
                                                                        leaf_size = value_matrix_2[j])

                for key, classifier in classifiers.items():
                    result = list()

                    avg_accuracy = 0.0

                    for train, test in cross_val:
                        train_feature_set = feature_set[train]
                        train_class_set = class_set[train]
                        test_feature_set = feature_set[test]
                        test_class_set = class_set[test]

                        classifier.fit(train_feature_set, train_class_set.ravel())

                        prediction = classifier.predict(test_feature_set)

                        test_class_set = test_class_set.astype('float')
                        prediction = prediction.astype('float')

                        result.append([test_class_set, prediction])

                    for r1 in range(len(result)):
                        avg_accuracy += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

                    avg_accuracy /= float(len(result))
                    results[i][j] += avg_accuracy / len(data_sets) / len(classifiers)

    print("{0:5}".format(""))
    for i in value_matrix_2:
        print("{0:<5d}".format(i))
    print("\n")
    for i in range(len(value_matrix_1)):
        print("{0:>5d}".format(value_matrix_1[i]))
        for j in range(len(results[i])):
            print("{0:<5.3f}".format(results[i][j]))
        print("\n")

if __name__ == "__main__":
    experiment_2a()