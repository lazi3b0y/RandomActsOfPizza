from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_clf_parameters, print_clf_acc_table, print_wilcoxon, print_current_data_set
from utils.parse import parse_csv

import numpy
import sklearn
import warnings

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

    classifiers = {}

    value_matrix = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = numpy.zeros((len(value_matrix), len(value_matrix)))

    # The actual experiment begins here.
    for optimization_set in optimization_sets:  # Loop through all the sets that we chose for optimization.
        class_set, feature_set = parse_csv(optimization_set)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # The 10x10 Fold Cross Validation.
        cross_val = sklearn.cross_validation.KFold(n=n_elements,
                                                   n_folds=n_folds)

        print_current_data_set(optimization_set)

        for i in range(len(value_matrix)):
            for j in range(len(value_matrix)):  # Horizontal values in our table
                # Prints the current parameters for our classifiers (Decision Trees/Random Forests/KNeighbors).
                print_clf_parameters(max_depth=value_matrix[i],
                                     min_samples_leaf=value_matrix[j],
                                     n_estimators=value_matrix[j],
                                     n_neighbors=value_matrix[j],
                                     leaf_size=value_matrix[i])

                # classifiers["custom_random_forest"] = RandomForest(n_estimators=value_matrix[j],
                #                                                    max_depth=value_matrix[i])
                #
                # classifiers["sklearn_random_forest"] = RandomForestClassifier(n_estimators=value_matrix[j],
                #                                                               max_depth=value_matrix[i])

                # classifiers["sklearn_decision_tree"] = DecisionTreeClassifier(min_samples_leaf = value_matrix[j],
                #                                                               max_depth = value_matrix[i])

                # classifiers["custom_decision_tree"] = DecisionTree(min_samples_leaf = value_matrix[j],
                #                                                    max_depth = value_matrix[i])

                classifiers["sklearn_neighbors"] = KNeighborsClassifier(n_neighbors = value_matrix[j],
                                                                        leaf_size = value_matrix[i],
                                                                        algorithm = 'kd_tree')

                for key, classifier in classifiers.items():
                    result = list()

                    avg_accuracy = 0.0

                    for train, test in cross_val:
                        train_feature_set = feature_set[train]
                        train_class_set = class_set[train]
                        test_feature_set = feature_set[test]
                        test_class_set = class_set[test]

                        classifier.fit(X=train_feature_set,
                                       y=train_class_set.ravel())

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
                        # hori_label = "n_estimators" for random forest, "n_neighbors"
                        # for nearest neighbor and "min_samples_leaf" for decision tree
                        hori_label = "n_estimators")


def experiment_2_testing():
    test_sets = [
        "resources/binary_data_sets/haberman.csv",
        "resources/multi_data_sets/vehicle.csv",
    ]

    classifiers = {
        "custom_decision_tree": DecisionTree(min_samples_leaf = 10,
                                             max_depth = 50),
        "custom_random_forest": RandomForest(n_estimators = 50,
                                             max_depth = 40),
        "sklearn_decision_tree": DecisionTreeClassifier(min_samples_leaf = 10,
                                                        max_depth = 50),
        "sklearn_random_forest": RandomForestClassifier(n_estimators = 50,
                                                        max_depth = 40),
        "sklearn_neighbors": KNeighborsClassifier(n_neighbors = 20,
                                                  leaf_size = 20,
                                                  algorithm = 'kd_tree'),
    }

    accuracies = {
        "custom_decision_tree": [0.0],
        "custom_random_forest": [0.0],
        "sklearn_decision_tree": [0.0],
        "sklearn_random_forest": [0.0],
        "sklearn_neighbors": [0.0],
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

        print_current_data_set(test_set)

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

            for row in range(len(result)):
                avg_accuracy += sklearn.metrics.accuracy_score(y_true = result[row][0],
                                                               y_pred = result[row][1])

            avg_accuracy /= float(len(result))

            accuracies[key][0] += avg_accuracy

    for key in accuracies:
        accuracies[key][0] /= float(len(test_sets))

    print_wilcoxon(accuracies)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # experiment_2_optimization()
    experiment_2_testing()
