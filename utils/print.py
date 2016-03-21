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
        print("\t\t{} {}".format("Spearman: ", scipy.stats.spearmanr(x, y)))
        print("\t\t{} {}\n".format("Pearson:  ", scipy.stats.pearsonr(x, y)))


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
    print("\n##############################")
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
    print("############################################################################")
    print("Wilcoxon Result for Decision Tree: ")
    print(wilcoxon(x = predictions['custom_decision_tree'],
                   y = predictions['sklearn_decision_tree']))
    print("Wilcoxon Result for Random Forest: ")
    print(wilcoxon(x = predictions['custom_random_forest'],
                   y = predictions['sklearn_random_forest']))
    if predictions['sklearn_neighbors']:
        print(wilcoxon(x = predictions['sklearn_neighbors']))
    print("############################################################################\n\n")


def print_clf_parameters(max_depth, min_samples_leaf, n_estimators, n_neighbors, leaf_size):
    print("##############################")
    print("\tClassifier parameters")
    print("##############################")
    print("Max depth:\t\t\t\t{}".format(max_depth))
    print("Minimum Samples Leaf:\t{}".format(min_samples_leaf))
    print("Number of estimators:\t{}".format(n_estimators))
    print("Number of neighbors:\t{}".format(n_neighbors))
    print("Leaf size:\t\t\t\t{}\n".format(leaf_size))


def print_current_data_set(path):
    print("############################################################################")
    print("\tData set: {}".format(path))
    print("############################################################################")


def print_clf_acc_table(hori_label, hori_values, vert_values, values):
    print("\t\t{}".format(hori_label))
    print("\t\t", end="")
    for i in hori_values:
        print("{0:>5d}\t\t".format(i), end="")
    print("")
    for i in range(len(vert_values)):
        print("{0:>5d}\t".format(vert_values[i]), end="")
        for j in range(len(values[i])):
            print("{0:5f}\t".format(values[i][j]), end="")
        print("")
