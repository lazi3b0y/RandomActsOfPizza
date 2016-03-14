from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy
import sklearn

__author__ = 'Simon & Oskar'


def experiment_2a():
    multiDataSet = ["multi/iris.csv", "multi/glass.csv", "multi/vehicle.csv",
                    "datasets/balance-scale.csv", "datasets/breast-cancer.csv",
                     "datasets/breast-w.csv", "datasets/credit-a.csv",
                     "datasets/credit-g.csv", "datasets/diabetes.csv",
                     "datasets/haberman.csv", "datasets/heart-c.csv",
                     "datasets/heart-h.csv", "datasets/heart-s.csv"]

    max_depth = 30 #20 - 70, 10
    min_samples_leaf = 1 #1 - 9, 2
    n_estimators = 50 #10 - 60, 10
    max_features = None

    folds = 2

    print("{}".format("Max depth:"))
    print("{}".format(max_depth))
    print("{}".format("Min_sample_leaf:"))
    print("{}".format(min_samples_leaf))
    print("{}".format("Number of estimators:"))
    print("{}".format(n_estimators))
    print("{}".format("Number of folds:"))
    print("{}".format(folds))

    custom_decision_tree = DecisionTree(max_depth = max_depth, min_samples_leaf = min_samples_leaf, max_features = max_features)
    sklearn_decision_tree = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, max_features = max_features)

    custom_random_forest = RandomForest(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = 30)
    sklearn_random_forest = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = 30, max_features = max_features)

    kNN = KNeighborsClassifier(n_neighbors=5, leaf_size=30)
    
    classifiers = {
        "custom_decision_tree": custom_decision_tree,
        "custom_random_forest": custom_random_forest,
        "sklearn_decision_tree": sklearn_decision_tree,
        "sklearn_random_forest": sklearn_random_forest
    }
    # classifiers["kNN"] = kNN

    maxD = [d for d in range(5, 50+1, 5)]
    minL = [d for d in range(30, 100+1, 10)]
    test2 = numpy.zeros((len(maxD), len(minL)))
    for data in multiDataSet: #binaryDataSet or multiDataSet
        set, classSet = HelpFunctions.extract_data_from_csv(data)

        kf = sklearn.cross_validation.KFold(set.shape[0], n_folds=folds)
        print("\n{0}{1}".format("DataSet: ", data))


        for i in range(len(maxD)):
            for j in range(len(minL)):
                # classifiers["OurRF"] = randomForest(n_estimators=minL[j], max_depth=maxD[i])
                # classifiers["SkRF"] = RandomForestClassifier(n_estimators=minL[j], max_depth=maxD[i])
                classifiers["kNN"] = KNeighborsClassifier(n_neighbors=maxD[i], leaf_size=minL[j])
                for k, v in classifiers.items():
                    result = list()

                    averageAcc = 0.0

                    for train, test in kf:
                        trainSet = set[train]
                        trainClass = classSet[train]
                        testSet = set[test]
                        testClass = classSet[test]

                        v.fit(trainSet, trainClass.ravel())

                        p = v.predict(testSet)

                        result.append(HelpFunctions.convert_pred_and_class_sets_to_values(testClass, p))

                    for r1 in range(len(result)):
                        averageAcc += sklearn.metrics.accuracy_score(result[r1][0], result[r1][1])

                    averageAcc = averageAcc / float(len(result))
                    test2[i][j] += averageAcc / len(multiDataSet) / len(classifiers)

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