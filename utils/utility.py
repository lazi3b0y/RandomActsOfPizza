import numpy

__author__ = 'Simon & Oskar'


def convert_strings_to_numeric(string_set):
    string_set = string_set.ravel()

    uniq_values = numpy.unique(string_set)

    a = numpy.empty(len(string_set))

    for i in range(string_set.shape[0]):
        for j in range(uniq_values.shape[0]):
            if string_set[i] == uniq_values[j]:
                a[i] = j
                break

    a = numpy.array(a).astype(numpy.float)

    return a.reshape((len(a), 1))
