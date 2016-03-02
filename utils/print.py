import scipy.stats
from pandas import read_csv
from utils.json_parser import parse_json

__author__ = 'Simon & Oskar'

def print_csv(csv_data):
    print(csv_data)  # pandas data-frame
    print(csv_data.values)  # numpy-array containing values

def print_spearman_and_pearson(csv_data):
    for column in csv_data.columns[:-1]:  # all columns except last
        x = csv_data[column]
        y = csv_data['received_pizza']

        print('{}{}'.format(column, ': '))
        print('     {} {}'.format('Spearman: ', scipy.stats.spearmanr(x, y)))
        print('     {} {}'.format('Pearson:  ', scipy.stats.pearsonr(x, y)))

if __name__ == '__main__':
    parse_json()
    csv_data = read_csv('resources/raop.csv')

    print_csv(csv_data)
    print_spearman_and_pearson(csv_data)
