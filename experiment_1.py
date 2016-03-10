from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from numpy import unique, array, concatenate, asfarray
from time import time
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from utils.parse import csv, json

__author__ = 'Simon & Oskar'

# TODO: Restructure the code somewhat to make it look less like JJJ's. Ayyyy lmao Kappa 123
def experiment():
    json()
    csv_data = csv('resources/raop.csv')

    class_set = csv_data.as_matrix(columns = csv_data.columns[-1:])
    class_set = asfarray(class_set)

    feature_set = csv_data.as_matrix(columns = csv_data.columns[1:])
    feature_set = asfarray(feature_set)
    print("Random guessing value: {}".format((1.0 / float(unique(class_set).shape[0]))))

    classifiers = {
        'custom_decision_tree': DecisionTree(max_depth=10, min_samples_leaf=30),
        'custom_random_forest': RandomForest(),
        'sklearn_decision_tree': DecisionTreeClassifier(),
        'sklearn_random_forest': RandomForestClassifier(),
    }

    folds = 2
    kf = cross_validation.KFold(feature_set.shape[0], n_folds=folds)
    predictions = {}
    for label, classifier in classifiers.items():
        result = list()
        prob = list()

        # TODO: Obvious fuckin' plagiarism shit, rename pl0x or find a way to remove the need to initialize them.
        avg_accuracy = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        avg_auc = 0.0
        avg_train_time = 0.0
        avg_test_time = 0.0

        print(label)
        for train, test in kf:
            train_feature_set = feature_set[train]
            train_class_set = class_set[train]
            test_feature_set = feature_set[test]
            test_class_set = class_set[test]

            start = time()
            classifier.fit(train_feature_set, train_class_set.ravel())

            # If the current classifier is our own decision tree,
            # print a visualization of it in the console.
            # if label == 'custom_decision_tree':
                # classifier.print()

            avg_train_time += time() - start

            start = time()
            prediction_result = classifier.predict(test_feature_set)
            prob.append(classifier.predict_proba(test_feature_set))
            avg_test_time += time() - start

            # TODO: change this section down to the result.append() function call. Rename variables etc.
            a = test_class_set.ravel()
            b = prediction_result.ravel()

            u = unique(concatenate((a, b)))

            for i in range(test_class_set.shape[0]):
                for j in range(u.shape[0]):
                    if test_class_set[i] == u[j]:
                        a[i] = j
                    if prediction_result[i] == u[j]:
                        b[i] = j

            a = a.astype('float')
            b = b.astype('float')


            result.append([a, b])

        # TODO: In need of some refactoring, names and structure needs to be redone.
        predictions[label] = list()
        for r1 in range(len(result)):
            avg_accuracy += metrics.accuracy_score(result[r1][0], result[r1][1])
            predictions[label].append(metrics.accuracy_score(result[r1][0], result[r1][1]))
            avg_precision += metrics.precision_score(result[r1][0], result[r1][1])
            avg_recall += metrics.recall_score(result[r1][0], result[r1][1])

            u = unique(result[r1][0])
            p1 = array(prob[r1])
            maxP1 = array([max(d) for d in p1])
            if len(u) > 2:
                for i in u:
                    fpr, tpr, thresholds = metrics.roc_curve(result[r1][0], maxP1, pos_label=int(i))
                    avg_auc += metrics.auc(fpr, tpr)
                avg_auc = avg_auc / (u.size / 2.0)
            else:
                fpr, tpr, thresholds = metrics.roc_curve(result[r1][0], maxP1)
                avg_auc += metrics.auc(fpr, tpr)

        # TODO: Move this crap to the print.py module.
        print("Average over {} folds".format(folds))
        print('Accuracy: {0:.3f}'.format(avg_accuracy / float(len(result))))
        print("Precision: {0:.3f}".format(avg_precision / float(len(result))))
        print("Recall: {0:.3f}".format(avg_recall / float(len(result))))
        print("Auc: {0:.3f}".format(avg_auc / float(len(result))))
        print('Training time: {}'.format((avg_train_time / float(len(result)))))
        print('Test time: {}'.format((avg_test_time / float(len(result)))))
        print("--------------")

    # TODO: This needs to be moved as well.
    print("Wilcoxon Result for Decision Tree: ")
    print(wilcoxon(predictions['custom_decision_tree'], predictions['sklearn_decision_tree']))
    print("Wilcoxon Result for Random Forest: ")
    print(wilcoxon(predictions['custom_random_forest'], predictions['sklearn_random_forest']))


if __name__ == "__main__":
    experiment()