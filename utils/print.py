import scipy.stats
from numpy import array_str
from collections import Counter
from utils.parse import Parse
from classifiers.decision_tree import DecisionTree

__author__ = 'Simon & Oskar'

class Print:
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

    def tree(self, node, depth):
        if type(node) is type(DecisionTree()):
            v = "+--"
            for i in range(depth):
                v = "|  " + v
            v += " " + array_str(node.value)
            print(v)
            self.print_tree(node.leftChild, depth + 1)
            self.print_tree(node.rightChild, depth + 1)
        else:
            v = "+-"
            for i in range(depth):
                v = "|  " + v
            v += "["
            v += str(Counter(node).most_common(2)) + ", "
            v = v[:-2] + "]"
            print(v)

    if __name__ == '__main__':
        Parse.json()
        csv_data = Parse.csv('resources/raop.csv')

        csv(csv_data)
        spearman_and_pearson(csv_data)
