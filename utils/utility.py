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


def convert_to_binary_numbers(class_set, diff):
    class_set = class_set.ravel()

    for i in range(class_set.shape[0]):
        class_set[i] = class_set[i] - diff

    return class_set.reshape((len(class_set), 1))
