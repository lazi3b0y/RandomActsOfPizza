import scipy.stats
from numpy import array_str
from collections import Counter
from utils import parse

__author__ = 'Simon & Oskar'



def csv(csv_data):
    print(csv_data)  # pandas data-frame
    print(csv_data.values)  # numpy-array containing values

def spearman_and_pearson(data):
    for column in data.columns[:-1]:  # all columns except last
        x = data[column]
        y = data['received_pizza']

        print('{}: '.format(column))
        print('     {} {}'.format('Spearman: ', scipy.stats.spearmanr(x, y)))
        print('     {} {}\n'.format('Pearson:  ', scipy.stats.pearsonr(x, y)))

def tree(node, depth):
    from classifiers.decision_tree import DecisionTree
    if type(node) is type(DecisionTree()):
        v = "+--"
        for i in range(depth):
            v = "|  " + v
        v += " " + array_str(node.value)
        print(v)
        tree(node.left_child, depth + 1)
        tree(node.right_child, depth + 1)
    else:
        v = "+-"
        for i in range(depth):
            v = "|  " + v
        v += "["
        v += str(Counter(node).most_common(2)) + ", "
        v = v[:-2] + "]"
        print(v)
