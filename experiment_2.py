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
    multiDataSet = ["resources/iris.csv", "resources/glass.csv", "resources/vehicle.csv",
                    "resources/balance-scale.csv", "resources/breast-cancer.csv",
                    "resources/breast-w.csv", "resources/credit-a.csv",
                    "resources/credit-g.csv", "resources/diabetes.csv",
                    "resources/haberman.csv", "resources/heart-c.csv",
                    "resources/heart-h.csv", "resources/heart-s.csv"]

    # 'Decision Tree'/'Random Forest' parameters.
    max_depth = 30
    min_samples_leaf = 1
    n_estimators = 50
    max_features = None

    # 'KNeighbors' parameters
    n_neighbors = 5
    leaf_size = 30

    # Prints the current parameters for our classifiers (Decision Trees and Random Forests).
    print_clf_parameters(max_depth = max_depth,
                         min_samples_leaf = min_samples_leaf,
                         n_estimators = n_estimators)

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
    
    classifiers = {
        "custom_decision_tree": custom_decision_tree,
        "custom_random_forest": custom_random_forest,
        "sklearn_decision_tree": sklearn_decision_tree,
        "sklearn_random_forest": sklearn_random_forest,
        "sklearn_neighbors": sklearn_neighbors,
    }

    maxD = [d for d in range(5, 51, 5)]
    minL = [d for d in range(30, 101, 10)]
    test2 = numpy.zeros((len(maxD), len(minL)))
    for data in multiDataSet:  # binaryDataSet or multiDataSet
        feature_set, class_set = parse_csv(data)

        # 'Cross Validation' parameters
        n_folds = 10
        n_elements = feature_set.shape[0]

        # This performs our 10x10 Fold Cross Validation.
        cross_val = sklearn.cross_validation.KFold(n = n_elements,
                                                   n_folds = n_folds)

        print("\n{0}{1}".format("DataSet: ", data))

        for i in range(len(maxD)):
            for j in range(len(minL)):
                # classifiers["OurRF"] = randomForest(n_estimators=minL[j], max_depth=maxD[i])
                # classifiers["SkRF"] = RandomForestClassifier(n_estimators=minL[j], max_depth=maxD[i])
                classifiers["sklearn_neighbors"] = KNeighborsClassifier(n_neighbors = maxD[i],
                                                                        leaf_size = minL[j])
                for k, v in classifiers.items():
                    result = list()

                    avg_accuracy = 0.0

                    for train, test in cross_val:
                        train_feature_set = feature_set[train]
                        train_class_set = class_set[train]
                        test_feature_set = feature_set[test]
                        test_class_set = class_set[test]

                        v.fit(train_feature_set, train_class_set.ravel())

                        prediction = v.predict(test_feature_set)

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

                    for r1 in range(len(result)):
                        avg_accuracy += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

                    avg_accuracy = avg_accuracy / float(len(result))
                    test2[i][j] += avg_accuracy / len(multiDataSet) / len(classifiers)

    print("{0:5}".format(""))
    for i in minL:
        print("{0:<5d}".format(i))
    print("\n")
    for i in range(len(maxD)):
        print("{0:>5d}".format(maxD[i]))
        for j in range(len(test2[i])):
            print("{0:<5.3f}".format(test2[i][j]))
        print("\n")

if __name__ == "__main__":
    experiment_2a()