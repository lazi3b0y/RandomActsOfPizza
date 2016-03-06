from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from numpy import unique, array
from time import time
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from utils.parse import Parse

__author__ = 'Simon & Oskar'

def experiment():

    set, classSet = Parse.csv('datasets/raop.csv')

    print("Random guessing value: {}".format((1.0 / float(unique(classSet).shape[0]))))

    ourDT = DecisionTree(max_depth=10, min_samples_leaf=30)
    skDT = DecisionTreeClassifier()

    ourRF = RandomForest()
    skRF = RandomForestClassifier()

    classifiers = {'ourDT': ourDT,
                   'ourRF': ourRF,
                   'skDT': skDT,
                   'skRF': skRF}

    folds = 2
    printed = False
    kf = cross_validation.KFold(set.shape[0], n_folds=folds)
    predictions = {}
    for k, v in classifiers.items():
        result = list()
        prob = list()

        avg_accuracy = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        avg_auc = 0.0
        avg_train_time = 0.0
        avg_test_time = 0.0

        print(k)
        for train, test in kf:
            train_set = set[train]
            train_class = classSet[train]
            test_set = set[test]
            test_class = classSet[test]

            start = time()
            v.fit(train_set, train_class.ravel())

            if k == 'ourDT' and printed == False:
                v.printTree() # Prints a tree
                printed = True

            avg_train_time += time() - start

            start = time()
            p = v.predict(test_set)
            prob.append(v.predict_proba(test_set))
            avg_test_time += time() - start

            result.append(HelpFunctions.convert_pred_and_class_sets_to_values(test_class, p))

        predictions[k] = list()
        for r1 in range(len(result)):
            avg_accuracy += metrics.accuracy_score(result[r1][0], result[r1][1])
            predictions[k].append(metrics.accuracy_score(result[r1][0], result[r1][1]))
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

        print("Average over {} folds".format(folds))
        print('Accuracy: {0:.3f}'.format(avg_accuracy / float(len(result))))
        print("Precision: {0:.3f}".format(avg_precision / float(len(result))))
        print("Recall: {0:.3f}".format(avg_recall / float(len(result))))
        print("Auc: {0:.3f}".format(avg_auc / float(len(result))))
        print('Training time: {}'.format((avg_train_time / float(len(result)))))
        print('Test time: {}'.format((avg_test_time / float(len(result)))))
        print("--------------")

    print("Wilcoxon DT: ")
    print(wilcoxon(predictions['ourDT'], predictions['skDT']))
    print("Wilcoxon RF: ")
    print(wilcoxon(predictions['ourRF'], predictions['skRF']))


if __name__ == "__main__":
    experiment()