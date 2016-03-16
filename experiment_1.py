from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from utils.parse import parse_json, parse_csv
from utils.print import print_label, print_statistics, print_wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time

import numpy
import sklearn


def experiment_1():
    # Relative file paths to the .json file and .csv file.
    json_path = 'resources/train.json'
    csv_path = 'resources/multi_data_sets/vehicle.csv'

    # Check if the current data set is binary or multi to
    # know what value average should have with some of
    # the sklearn.metric methods later on.
    index = csv_path.find('binary_data_set')
    if index >= 0:  # If the string 'binary_data_set' is found in the current path to the .csv file.
        average = "binary"
    else:  # If the string 'binary_data_set' was not found.
        average = "weighted"

    # Load and parse the json file, then save the parsed data
    # to a .csv file for later use. If raop.csv already exists
    # this can be safetly ignored and left commented.
    # parse_json(json_path, csv_path)

    # Load and parse the csv. Returns two numpy arrays.
    # One containing the classes and one with the features.
    class_set, feature_set = parse_csv(csv_path)

    # 'Decision Tree'/'Random Forest' parameters.
    max_depth = 10
    min_samples_leaf = 30
    max_features = None

    # 'Cross Validation' parameters
    n_folds = 10
    n_elements = feature_set.shape[0]

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
    custom_random_forest = RandomForest(max_depth = max_depth,
                                        min_samples_leaf = min_samples_leaf,
                                        max_features = max_features)

    sklearn_random_forest = RandomForestClassifier(max_depth = max_depth,
                                                   min_samples_leaf = min_samples_leaf,
                                                   max_features = max_features)

    # A dictionary containing our classifiers that were initialized before.
    classifiers = {
        "custom_decision_tree": custom_decision_tree,
        "custom_random_forest": custom_random_forest,
        "sklearn_decision_tree": sklearn_decision_tree,
        "sklearn_random_forest": sklearn_random_forest,
    }

    # Initialization of the dictionary where our predictions will be stored.
    predictions = {
        "custom_decision_tree": list(),
        "custom_random_forest": list(),
        "sklearn_decision_tree": list(),
        "sklearn_random_forest": list(),
    }

    # The 10x10 Fold Cross Validation, divides the data set into training set and test set.
    # In our case we'll end up with a test set that 1/10 of the total data and a train set
    # thats 9/10 of the total data. Atleast as long as n_folds = 10.
    cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                               n_folds = n_folds)

    for key, classifier in classifiers.items():
        result = list()
        pred_pbty = list()

        avg_accuracy = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        avg_auc = 0.0
        avg_train_time = 0.0
        avg_test_time = 0.0

        print_label(key)
        for train, test in cross_val:
            train_feature_set = feature_set[train]
            train_class_set = class_set[train]
            test_feature_set = feature_set[test]
            test_class_set = class_set[test]

            start = time()
            classifier.fit(numpy.array(train_feature_set), numpy.array(train_class_set.ravel()))

            avg_train_time += time() - start

            start = time()
            prediction = classifier.predict(numpy.array(test_feature_set))
            pred_pbty.append(classifier.predict_proba(numpy.array(test_feature_set)))
            avg_test_time += time() - start

            test_class_set = test_class_set.astype('float')
            prediction = prediction.astype('float')

            result.append([test_class_set, prediction])

        for row in range(len(result)):
            avg_accuracy += sklearn.metrics.accuracy_score(result[row][0], result[row][1])
            predictions[key].append(sklearn.metrics.precision_score(result[row][0], result[row][1], average = average))
            avg_precision += sklearn.metrics.precision_score(result[row][0], result[row][1], average = average)
            avg_recall += sklearn.metrics.recall_score(result[row][0], result[row][1], average = average)

            uniq_values = numpy.unique(result[row][0])
            pred_pbty_sub_set = numpy.array(pred_pbty[row])
            maximized_pbty_sub_set = numpy.array([max(pbty_pair) for pbty_pair in pred_pbty_sub_set])

            if len(uniq_values) > 2:
                for value in uniq_values:
                    false_posi_rates, true_posi_rates, thresholds = sklearn.metrics.roc_curve(result[row][0], maximized_pbty_sub_set, int(value))
                    avg_auc += sklearn.metrics.auc(false_posi_rates, true_posi_rates)

                avg_auc /= uniq_values.size / 2.0
            else:
                false_posi_rates, true_posi_rates, thresholds = sklearn.metrics.roc_curve(result[row][0], maximized_pbty_sub_set)
                avg_auc += sklearn.metrics.auc(false_posi_rates, true_posi_rates)

        print_statistics(avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result)

    print_wilcoxon(predictions)


if __name__ == "__main__":
    experiment_1()
