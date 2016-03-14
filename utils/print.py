import scipy.stats
from numpy import array_str
from collections import Counter
from scipy.stats import wilcoxon

__author__ = 'Simon & Oskar'


def print_csv(csv_data):
    print(csv_data)  # pandas data-frame
    print(csv_data.values)  # numpy-array containing values


def print_spearman_and_pearson(data):
    for column in data.columns[:-1]:  # all columns except last
        x = data[column]
        y = data['received_pizza']

        print('{}: '.format(column))
        print('     {} {}'.format('Spearman: ', scipy.stats.spearmanr(x, y)))
        print('     {} {}\n'.format('Pearson:  ', scipy.stats.pearsonr(x, y)))


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
    print("     {}".format(label))
    print("##############################")


def print_statistics(folds, avg_accuracy, avg_precision, avg_recall, avg_auc, avg_train_time, avg_test_time, result):
    print("Average over {} folds".format(folds))
    print('Accuracy: {0:.3f}'.format(avg_accuracy / float(len(result))))
    print("Precision: {0:.3f}".format(avg_precision / float(len(result))))
    print("Recall: {0:.3f}".format(avg_recall / float(len(result))))
    print("Auc: {0:.3f}".format(avg_auc / float(len(result))))
    print('Training time: {}'.format((avg_train_time / float(len(result)))))
    print('Test time: {}'.format((avg_test_time / float(len(result)))))


def print_wilcoxon(predictions):
    print("\n")
    print("##############################################")
    print("Wilcoxon Result for Decision Tree: ")
    print(wilcoxon(x = predictions['custom_decision_tree'], y = predictions['sklearn_decision_tree']))
    print("Wilcoxon Result for Random Forest: ")
    print(wilcoxon(x = predictions['custom_random_forest'], y = predictions['sklearn_random_forest']))
    print("##############################################") 