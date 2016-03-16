import numpy

__author__ = 'Simon & Oskar'


def convert_strings_to_numeric(class_set, prediction_set):
    class_set = class_set.ravel()
    prediction_set = prediction_set.ravel()

    a = numpy.zeros(len(class_set))
    b = numpy.zeros(len(prediction_set))

    uniq_values = numpy.unique(numpy.concatenate((class_set, prediction_set)))

    for i in range(class_set.shape[0]):
        for j in range(uniq_values.shape[0]):
            if class_set[i] == uniq_values[j]:
                a[i] = j
            if prediction_set[i] == uniq_values[j]:
                b[i] = j

    a = numpy.array(a).astype(numpy.float)
    b = numpy.array(b).astype(numpy.float)

    return a, b