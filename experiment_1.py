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
    csv_data = parse_csv(csv_path)

    classSet = csv_data.as_matrix(columns = csv_data.columns[-1:])
    classSet = numpy.asfarray(classSet)

    set = csv_data.as_matrix(columns = csv_data.columns[1:])
    set = numpy.asfarray(set)

    print("Random guessing value: {}".format((1.0 / float(numpy.unique(classSet).shape[0]))))

    ourDT = DecisionTree(max_depth=10, min_samples_leaf=30)
    skDT = DecisionTreeClassifier()

    ourRF = RandomForest()
    skRF = RandomForestClassifier()

    cTypes = {}
    cTypes["custom_decision_tree"] = ourDT
    cTypes["custom_random_forest"] = ourRF
    cTypes["sklearn_decision_tree"] = skDT
    cTypes["sklearn_random_forest"] = skRF

    folds = 10
    printed = False
    kf = cross_validation.KFold(set.shape[0], n_folds=folds)
    predictions = {}
    for k, v in cTypes.items():
        result = list()
        prob = list()

        averageAcc = 0.0
        avgPre = 0.0
        avgRecall = 0.0
        avgAuc = 0.0
        avgTrainTime = 0.0
        avgTestTime = 0.0

        print_label(k)
        for train, test in kf:
            trainSet = set[train]
            trainClass = classSet[train]
            testSet = set[test]
            testClass = classSet[test]

            start = time()
            v.fit(trainSet, trainClass.ravel())

            if k == 'ourDT' and printed == False:
                # v.printTree() # Prints a tree
                printed = True

            avgTrainTime += time() - start

            start = time()
            pred = v.predict(testSet)
            prob.append(v.predict_proba(testSet))
            avgTestTime += time() - start

            trueSet = numpy.array([c[0] for c in testClass])
            predSet = numpy.array([p for p in pred])

            u = numpy.unique(numpy.concatenate((trueSet, predSet)))

            for i in range(testClass.shape[0]):
                for j in range(u.shape[0]):
                    if testClass[i] == u[j]:
                        trueSet[i] = j
                    if pred[i] == u[j]:
                        predSet[i] = j

            trueSet = trueSet.astype('float')
            predSet = predSet.astype('float')

            result.append([trueSet, predSet])

        predictions[k] = list()
        for r1 in range(len(result)):
            averageAcc += metrics.accuracy_score(result[r1][0], result[r1][1])
            predictions[k].append(metrics.accuracy_score(result[r1][0], result[r1][1]))
            avgPre += metrics.precision_score(result[r1][0], result[r1][1])
            avgRecall += metrics.recall_score(result[r1][0], result[r1][1])

            u = numpy.unique(result[r1][0])
            p1 = numpy.array(prob[r1])
            maxP1 = numpy.array([max(d) for d in p1])
            if len(u) > 2:
                for i in u:
                    fpr, tpr, thresholds = metrics.roc_curve(result[r1][0], maxP1, pos_label=int(i))
                    avgAuc += metrics.auc(fpr, tpr)
                avgAuc = avgAuc / (u.size / 2.0)
            else:
                fpr, tpr, thresholds = metrics.roc_curve(result[r1][0], maxP1)
                avgAuc += metrics.auc(fpr, tpr)
        print_statistics(folds, averageAcc, avgPre, avgRecall, avgAuc, avgTrainTime, avgTestTime, result)

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
#     folds = 10
#     printed = False
#     kf = sklearn.cross_validation.KFold(feature_set.shape[0], n_folds=folds)
#     predictions = {}
#     for label, classifier in classifiers.items():
#         result = list()
#         prob = list()
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
#             trainSet = feature_set[train]
#             trainClass = class_set[train]
#             test_feature_set = feature_set[test]
#             test_class_set = class_set[test]
#
#             start = time()
#             classifier.fit(trainSet, trainClass.ravel())
#
#             if label == 'custom_decision_tree' and printed == False:
#                 # v.printTree() # Prints a tree
#                 printed = True
#
#             avg_train_time += time() - start
#
#             start = time()
#             p = classifier.predict(test_feature_set)
#             prob.append(classifier.predict_proba(test_feature_set))
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
#             p1 = numpy.array(prob[i])
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
#         print_statistics(folds, avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result)
#
#     print_wilcoxon(predictions)
#
#
# if __name__ == "__main__":
#     experiment()