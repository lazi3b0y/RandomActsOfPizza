from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.print import print_clf_parameters, print_clf_acc_table
from utils.parse import parse_csv

import numpy
import sklearn

__author__ = "Simon & Oskar"


def experiment_2():
    optimization_sets = [
        "resources/binary_data_sets/balance-scale.csv",
        "resources/binary_data_sets/breast-cancer.csv",
        "resources/binary_data_sets/diabetes.csv",
        "resources/multi_data_sets/glass.csv",
        "resources/multi_data_sets/iris.csv",
        "resources/multi_data_sets/splice.csv",
    ]

    test_sets = [
        "resources/binary_data_sets/haberman.csv",
        "resources/multi_data_sets/vehicle.csv",
    ]

    # A dictionary containing our classifiers that were initialized before.
    classifiers = {
        # "custom_decision_tree": None,
        # "custom_random_forest": None,
        # "sklearn_decision_tree": None,
        # "sklearn_random_forest": None,
        "sklearn_neighbors": None,
    }

    value_matrix_1 = [v for v in range(5, 51, 5)]
    value_matrix_2 = [v for v in range(30, 101, 10)]
    results = numpy.zeros((len(value_matrix_1), len(value_matrix_2)))

    # The actual experiment begins here.
    for optimization_set in optimization_sets:
        class_set, feature_set = parse_csv(optimization_set)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # The 10x10 Fold Cross Validation, divides the data set into training set and test set.
        # In our case we'll end up with a test set that 1/10 of the total data and a train set
        # thats 9/10 of the total data. Atleast as long as n_folds = 10.
        cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                                   n_folds = n_folds)

        print("Current data set:\t{}".format(optimization_set))

        for i in range(len(value_matrix_1)):
            for j in range(len(value_matrix_2)):
                # Prints the current parameters for our classifiers (Decision Trees/Random Forests/KNeighbors).
                print_clf_parameters(max_depth = value_matrix_1[i],
                                     min_samples_leaf = 1,
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

                        test_class_set = test_class_set.astype(numpy.float)
                        prediction = prediction.astype(numpy.float)

                        result.append([test_class_set, prediction])

                    for r1 in range(len(result)):
                        avg_accuracy += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

                    avg_accuracy /= float(len(result))
                    results[i][j] += avg_accuracy / len(optimization_sets) / len(classifiers)

    print_clf_acc_table(value_matrix_2, value_matrix_1, results)

if __name__ == "__main__":
    experiment_2()
