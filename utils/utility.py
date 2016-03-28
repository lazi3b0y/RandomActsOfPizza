from scipy.stats import binom
from random import randrange

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

    a = a.astype(numpy.float)

    return a.reshape((len(a), 1))


def convert_to_binary_numbers(class_set, diff):
    class_set = class_set.ravel()

    for i in range(class_set.shape[0]):
        class_set[i] = class_set[i] - diff

    return class_set.reshape((len(class_set), 1))


def mcnemar_midp(b, c):
    # Found at https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp


def pick_result(values):
    if len(values) != 1:
        eq = list()
        eq.append(0)
        for v in range(1, len(values)):
            if values[0][1] == values[v][1]:
                eq.append(v)
        return values[eq[randrange(0, len(eq))]][0]
    else:
        return values[0][0]
