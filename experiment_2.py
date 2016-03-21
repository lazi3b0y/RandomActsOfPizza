from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_clf_parameters, print_clf_acc_table, print_wilcoxon
from utils.parse import parse_csv
from scipy.stats import wilcoxon

import numpy
import sklearn

__author__ = "Simon & Oskar"


def experiment_2_optimization():
    optimization_sets = [
        "resources/binary_data_sets/balance-scale.csv",
        "resources/binary_data_sets/breast-cancer.csv",
        "resources/binary_data_sets/diabetes.csv",
        "resources/multi_data_sets/glass.csv",
        "resources/multi_data_sets/iris.csv",
        "resources/multi_data_sets/splice.csv",
    ]

    classifiers = {
        # "custom_decision_tree": None,
        # "custom_random_forest": None,
        # "sklearn_decision_tree": None,
        # "sklearn_random_forest": None,
        "sklearn_neighbors": None,
    }

    value_matrix = [v for v in range(5, 51, 5)]
    results = numpy.zeros((len(value_matrix), len(value_matrix)))

    # The actual experiment begins here.
    for optimization_set in optimization_sets:
        class_set, feature_set = parse_csv(optimization_set)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # The 10x10 Fold Cross Validation.
        cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                                   n_folds = n_folds)

        print("Current data set:\t{}".format(optimization_set))

        for i in range(len(value_matrix)):
            for j in range(len(value_matrix)):  # Horizontal values in our table
                # Prints the current parameters for our classifiers (Decision Trees/Random Forests/KNeighbors).
                print_clf_parameters(max_depth = value_matrix[i],
                                     min_samples_leaf = value_matrix[j],
                                     n_estimators = value_matrix[j],
                                     n_neighbors = value_matrix[i],
                                     leaf_size = value_matrix[j])

                # classifiers["custom_random_forest"] = RandomForest(n_estimators = value_matrix[j],
                #                                                    max_depth = value_matrix[i])

                # classifiers["sklearn_random_forest"] = RandomForestClassifier(n_estimators = value_matrix[j],
                #                                                               max_depth = value_matrix[i])

                # classifiers["sklearn_decision_tree"] = DecisionTreeClassifier(n_neighbors = value_matrix[i],
                #                                                        leaf_size = value_matrix[j])

                # classifiers["custom_decision_tree"] = DecisionTree(n_neighbors = value_matrix[i],
                #                                                        leaf_size = value_matrix[j])

                # classifiers["sklearn_neighbors"] = KNeighborsClassifier(n_neighbors = value_matrix[i],
                #                                                         leaf_size = value_matrix[j])

                for key, classifier in classifiers.items():
                    result = list()

                    avg_accuracy = 0.0

                    for train, test in cross_val:
                        train_feature_set = feature_set[train]
                        train_class_set = class_set[train]
                        test_feature_set = feature_set[test]
                        test_class_set = class_set[test]

                        classifier.fit(X = train_feature_set,
                                       y = train_class_set.ravel())

                        prediction = classifier.predict(test_feature_set)

                        test_class_set = test_class_set.astype(numpy.float)
                        prediction = prediction.astype(numpy.float)

                        result.append([test_class_set, prediction])

                    for r1 in range(len(result)):
                        avg_accuracy += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

                    avg_accuracy /= float(len(result))
                    results[i][j] += avg_accuracy / len(optimization_sets) / len(classifiers)

    print_clf_acc_table(hori_values = value_matrix,
                        vert_values = value_matrix,
                        values = results,
                        hori_label = "n_estimators") # hori_label = "n_estimators" for random forest, "n_neighbors" for nearest neighbor and "min_samples_leaf" for decision tree


def experiment_2_testing():
    test_sets = [
        "resources/binary_data_sets/haberman.csv",
        "resources/multi_data_sets/vehicle.csv",
    ]

    classifiers = {
        "custom_decision_tree": None,
        "custom_random_forest": None,
        "sklearn_decision_tree": None,
        "sklearn_random_forest": None,
        "sklearn_neighbors": None,
    }

    predictions = {
        "custom_decision_tree": 0.0,
        "custom_random_forest": 0.0,
        "sklearn_decision_tree": 0.0,
        "sklearn_random_forest": 0.0,
        "sklearn_neighbors": 0.0,
    }

    # The actual experiment begins here.
    for test_set in test_sets:
        class_set, feature_set = parse_csv(test_set)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # The 10x10 Fold Cross Validation.
        cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                                   n_folds = n_folds)

        print("Current data set:\t{}".format(test_set))

        for key, classifier in classifiers.items():
            result = list()

            avg_accuracy = 0.0

            for train, test in cross_val:
                train_feature_set = feature_set[train]
                train_class_set = class_set[train]
                test_feature_set = feature_set[test]
                test_class_set = class_set[test]

                classifier.fit(X = train_feature_set,
                               y = train_class_set.ravel())

                prediction = classifier.predict(test_feature_set)

                test_class_set = test_class_set.astype(numpy.float)
                prediction = prediction.astype(numpy.float)

                result.append([test_class_set, prediction])

            for r1 in range(len(result)):
                avg_accuracy += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

            avg_accuracy /= float(len(result))

            predictions[key] += avg_accuracy

    for p in predictions:
        p /= len(test_sets)

    print_wilcoxon(predictions)

if __name__ == "__main__":
    experiment_2_optimization()
    # experiment_2_testing()
