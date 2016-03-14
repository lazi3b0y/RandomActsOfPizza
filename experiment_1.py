from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from utils.parse import parse_json, parse_csv
from utils.print import print_label, print_statistics, print_wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time

import numpy
import sklearn


def main():
    # json_path = 'resources/train.json'
    csv_path = 'resources/raop.csv'
    # parse_json(json_path, csv_path)
    class_set, feature_set = parse_csv(csv_path)

    n_folds = 10
    n_elements = feature_set.shape[0]
    max_depth = 10
    min_samples_leaf = 30
    max_features = None

    custom_decision_tree = DecisionTree(max_depth = max_depth,
                                        min_samples_leaf = min_samples_leaf)

    sklearn_decision_tree = DecisionTreeClassifier(max_depth = max_depth,
                                                   min_samples_leaf = min_samples_leaf,
                                                   max_features = max_features)

    custom_random_forest = RandomForest(max_depth = max_depth,
                                        min_samples_leaf = min_samples_leaf)

    sklearn_random_forest = RandomForestClassifier(max_depth = max_depth,
                                                   min_samples_leaf = min_samples_leaf,
                                                   max_features = max_features)

    classifiers = {
        "custom_decision_tree": custom_decision_tree,
        "custom_random_forest": custom_random_forest,
        "sklearn_decision_tree": sklearn_decision_tree,
        "sklearn_random_forest": sklearn_random_forest
    }

    cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                               n_folds = n_folds)
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
        for train, test in cross_val:
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
            avg_accuracy += sklearn.metrics.accuracy_score(result[row][0], result[row][1])
            predictions[key].append(sklearn.metrics.accuracy_score(result[row][0], result[row][1]))
            avg_precision += sklearn.metrics.precision_score(result[row][0], result[row][1])
            avg_recall += sklearn.metrics.recall_score(result[row][0], result[row][1])

            uniq_values = numpy.unique(result[row][0])
            pred_pbty_sub_set = numpy.array(pred_pbty[row])
            maximized_pbty_sub_set = numpy.array([max(pbty_pair) for pbty_pair in pred_pbty_sub_set])
            if len(uniq_values) > 2:
                for i in uniq_values:
                    false_posi_rates, true_posi_rates, thresholds = sklearn.metrics.roc_curve(y_true = result[row][0],
                                                                                              y_score = maximized_pbty_sub_set,
                                                                                              pos_label = int(i))

                    avg_auc += sklearn.metrics.auc(false_posi_rates, true_posi_rates)
                avg_auc /= uniq_values.size / 2.0
            else:
                false_posi_rates, true_posi_rates, thresholds = sklearn.metrics.roc_curve(result[row][0], maximized_pbty_sub_set)
                avg_auc += sklearn.metrics.auc(false_posi_rates, true_posi_rates)
        print_statistics(n_folds, avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result)

    print_wilcoxon(predictions)


if __name__ == "__main__":
    main()
