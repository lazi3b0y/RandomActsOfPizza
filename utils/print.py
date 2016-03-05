import scipy.stats
from utils.parser import parse_json, parse_csv

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

if __name__ == '__main__':
    parse_json()
    csv_data = parse_csv('resources/raop.csv')

    print_csv(csv_data)
    print_spearman_and_pearson(csv_data)
