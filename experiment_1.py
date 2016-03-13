from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from utils.parse import parse_json, parse_csv
from utils.print import print_label, print_statistics, print_wilcoxon

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
from time import time

import numpy


def main():
    json_path = 'resources/train.json'
    csv_path = 'resources/raop.csv'
    parse_json(json_path, csv_path)
    class_set, feature_set = parse_csv(csv_path)

    n_folds = 10
    n_elements = feature_set.shape[0]

    print("Random guessing value: {}".format((1.0 / float(numpy.unique(class_set).shape[0]))))

    custom_decision_tree = DecisionTree(max_depth=10, min_samples_leaf=30)
    sklearn_decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=30, max_features=None)

    custom_random_forest = RandomForest(max_depth=10, min_samples_leaf=30)
    sklearn_random_forest = RandomForestClassifier(max_depth=10, min_samples_leaf=30, max_features=None)

    classifiers = {}
    classifiers["custom_decision_tree"] = custom_decision_tree
    classifiers["custom_random_forest"] = custom_random_forest
    classifiers["sklearn_decision_tree"] = sklearn_decision_tree
    classifiers["sklearn_random_forest"] = sklearn_random_forest

    kf = cross_validation.KFold(n=n_elements, n_folds=n_folds)
    predictions = {}
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
        for train, test in kf:
            train_feature_set = feature_set[train]
            train_class_set = class_set[train]
            test_feature_set = feature_set[test]
            test_class_set = class_set[test]

            start = time()
            classifier.fit(train_feature_set, train_class_set.ravel())

            avg_train_time += time() - start

            start = time()
            prediction = classifier.predict(test_feature_set)
            pred_pbty.append(classifier.predict_proba(test_feature_set))
            avg_test_time += time() - start

            true_set = numpy.array([c[0] for c in test_class_set])
            pred_set = numpy.array([p for p in prediction])

            uniq_values = numpy.unique(numpy.concatenate((true_set, pred_set)))

            for i in range(test_class_set.shape[0]):
                for j in range(uniq_values.shape[0]):
                    if test_class_set[i] == uniq_values[j]:
                        true_set[i] = j
                    if prediction[i] == uniq_values[j]:
                        pred_set[i] = j

            true_set = true_set.astype('float32')
            pred_set = pred_set.astype('float32')

            result.append([true_set, pred_set])

        predictions[key] = list()
        for row in range(len(result)):
            avg_accuracy += metrics.accuracy_score(result[row][0], result[row][1])
            predictions[key].append(metrics.accuracy_score(result[row][0], result[row][1]))
            avg_precision += metrics.precision_score(result[row][0], result[row][1])
            avg_recall += metrics.recall_score(result[row][0], result[row][1])

            uniq_values = numpy.unique(result[row][0])
            pred_pbty_sub_set = numpy.array(pred_pbty[row])
            maximized_pbty_sub_set = numpy.array([max(pbty_pair) for pbty_pair in pred_pbty_sub_set])
            if len(uniq_values) > 2:
                for i in uniq_values:
                    false_pos_rates, true_pos_rates, thresholds = metrics.roc_curve(y_true = result[row][0], y_score = maximized_pbty_sub_set, pos_label = int(i))
                    avg_auc += metrics.auc(false_pos_rates, true_pos_rates)
                avg_auc /= uniq_values.size / 2.0
            else:
                false_pos_rates, true_pos_rates, thresholds = metrics.roc_curve(result[row][0], maximized_pbty_sub_set)
                avg_auc += metrics.auc(false_pos_rates, true_pos_rates)
        print_statistics(n_folds, avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result)

    print_wilcoxon(predictions)


if __name__ == "__main__":
    main()

    # # TODO: Restructure the code somewhat to make it look less like JJJ's. Ayyyy lmao Kappa 123
    # def experiment():
    #     json_path = 'resources/train.json'
    #     csv_path = 'resources/raop.csv'
    #     parse_json(json_path, csv_path)
    #     csv_data = parse_csv(csv_path)
    #
    #     class_set = csv_data.as_matrix(columns = csv_data.columns[-1:])
    #     class_set = numpy.asfarray(class_set)
    #
    #     feature_set = csv_data.as_matrix(columns = csv_data.columns[1:])
    #     feature_set = numpy.asfarray(feature_set)
    #     print("Random guessing value: {}\n".format((1.0 / float(numpy.unique(class_set).shape[0]))))
    #
    #     classifiers = {
    #         'custom_decision_tree': DecisionTree(max_depth=10, min_samples_leaf=30),
    #         'custom_random_forest': RandomForest(),
    #         'sklearn_decision_tree': sklearn.tree.DecisionTreeClassifier(),
    #         'sklearn_random_forest': sklearn.ensemble.RandomForestClassifier(),
    #     }
    #
    #     n_folds = 10
    #     printed = False
    #     kf = sklearn.cross_validation.KFold(feature_set.shape[0], n_folds=n_folds)
    #     predictions = {}
    #     for label, classifier in classifiers.items():
    #         result = list()
    #         pred_pbty = list()
    #
    #         avg_accuracy = 0.0
    #         avg_precision = 0.0
    #         avg_recall = 0.0
    #         avg_auc = 0.0
    #         avg_train_time = 0.0
    #         avg_test_time = 0.0
    #
    #         print(label)
    #         for train, test in kf:
    #             train_feature_set = feature_set[train]
    #             train_class_set = class_set[train]
    #             test_feature_set = feature_set[test]
    #             test_class_set = class_set[test]
    #
    #             start = time()
    #             classifier.fit(train_feature_set, train_class_set.ravel())
    #
    #             if label == 'custom_decision_tree' and printed == False:
    #                 # classifier.printTree() # Prints a tree
    #                 printed = True
    #
    #             avg_train_time += time() - start
    #
    #             start = time()
    #             p = classifier.predict(test_feature_set)
    #             pred_pbty.append(classifier.predict_proba(test_feature_set))
    #             avg_test_time += time() - start
    #
    #             # TODO: change this section down to the result.append() function call. Rename variables etc.
    #             flattened_class_set = test_class_set.ravel()
    #             flattened_prediction_result = p.ravel()
    #
    #             unique_values = numpy.unique(numpy.concatenate((flattened_class_set, flattened_prediction_result)))
    #
    #             for i in range(test_class_set.shape[0]):
    #                 for j in range(unique_values.shape[0]):
    #                     if test_class_set[i] == unique_values[j]:
    #                         flattened_class_set[i] = j
    #                     if p[i] == unique_values[j]:
    #                         flattened_class_set[i] = j
    #
    #             flattened_class_set = flattened_class_set.astype('float')
    #             flattened_prediction_result = flattened_prediction_result.astype('float')
    #
    #             result.append([flattened_class_set, flattened_prediction_result])
    #         # TODO: In need of some refactoring, names and structure needs to be redone.
    #         predictions[label] = list()
    #         for i in range(len(result)):
    #             avg_accuracy += sklearn.metrics.accuracy_score(result[i][0], result[i][1])
    #             predictions[label].append(sklearn.metrics.accuracy_score(result[i][0], result[i][1]))
    #             avg_precision += sklearn.metrics.precision_score(result[i][0], result[i][1])
    #             avg_recall += sklearn.metrics.recall_score(result[i][0], result[i][1])
    #
    #             unique_values = numpy.unique(result[i][0])
    #             p1 = numpy.array(pred_pbty[i])
    #             maxP1 = numpy.array([max(d) for d in p1])
    #             if len(unique_values) > 2:
    #                 for j in unique_values:
    #                     fpr, tpr, thresholds = sklearn.metrics.roc_curve(result[i][0], maxP1, pos_label=int(j))
    #                     avg_auc += sklearn.metrics.auc(fpr, tpr)
    #                 avg_auc /= unique_values.size / 2.0
    #             else:
    #                 fpr, tpr, thresholds = sklearn.metrics.roc_curve(result[i][0], maxP1)
    #                 avg_auc += sklearn.metrics.auc(fpr, tpr)
    #
    #         print_statistics(n_folds, avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result)
    #
    #     print_wilcoxon(predictions)
    #
    #
    # if __name__ == "__main__":
    #     experiment()
