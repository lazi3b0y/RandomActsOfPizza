import scipy.stats
from numpy import array_str
from collections import Counter
from scipy.stats import wilcoxon

__author__ = "Simon & Oskar"


def print_csv(csv_data):
    print(csv_data)  # pandas data-frame
    print(csv_data.values)  # numpy-array containing values


def print_spearman_and_pearson(data):
    for column in data.columns[:-1]:  # all columns except last
        x = data[column]
        y = data["received_pizza"]

        print("{}: ".format(column))
        print("     {} {}".format("Spearman: ", scipy.stats.spearmanr(x, y)))
        print("     {} {}\n".format("Pearson:  ", scipy.stats.pearsonr(x, y)))


def print_tree(root_node, depth):
    from classifiers.decision_tree import DecisionTree
    if isinstance(root_node, DecisionTree):
        v = "+--"
        for i in range(depth):
            v = "|  " + v
        v += " " + array_str(root_node.value)
        print(v)
        print_tree(root_node.left_child, depth + 1)
        print_tree(root_node.right_child, depth + 1)
    else:
        v = "+-"
        for i in range(depth):
            v = "|  " + v
        v += "["
        v += str(Counter(root_node).most_common(2)) + ", "
        v = v[:-2] + "]"
        print(v)


def print_label(label):
    print("\n")
    print("##############################")
    print("\t{}".format(label))
    print("##############################")


def print_statistics(avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result):
    print("Accuracy:\t\t{}".format(avg_accuracy / float(len(result))))
    print("Precision:\t\t{}".format(avg_precision / float(len(result))))
    print("Recall:\t\t\t{}".format(avg_recall / float(len(result))))
    print("Auc:\t\t\t{}".format(avg_auc / float(len(result))))
    print("Training time:\t{}".format((avg_train_time / float(len(result)))))
    print("Test time:\t\t{}".format((avg_test_time / float(len(result)))))


def print_wilcoxon(predictions):
    print("\n")
    print("##############################################")
    print("Wilcoxon Result for Decision Tree: ")
    print(wilcoxon(x = predictions['custom_decision_tree'], y = predictions['sklearn_decision_tree']))
    print("Wilcoxon Result for Random Forest: ")
    print(wilcoxon(x = predictions['custom_random_forest'], y = predictions['sklearn_random_forest']))
    print("##############################################")


def print_clf_parameters(max_depth, min_samples_leaf, n_estimators):
    print("##############################")
    print("\tClassifier parameters")
    print("##############################")
    print("{}".format("Max depth:"))
    print("{}".format(max_depth))
    print("{}".format("Min_sample_leaf:"))
    print("{}".format(min_samples_leaf))
    print("{}".format("Number of estimators:"))
    print("{}".format(n_estimators))