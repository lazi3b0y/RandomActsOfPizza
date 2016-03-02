__author__ = 'Simon & Oskar'
import pandas as ps
import numpy as np
import scipy.stats

def __init__(self, criterion = 'gini', max_feature = None, max_depth = None, min_samples_leaf = None, laplace = 0):
    self.criterion = criterion # gini or entropy
    self.max_feature = max_feature # random subspace size
    self.max_depth = max_depth  # maximum leaf depth
    self.min_samples_leaf = min_samples_leaf # minimum leaf size
    self.laplace = laplace # laplace correction for probability estimation

def fit(X, y): # train model

    """
def predict(X): # classify objects


def predict_proba(X): # class probability estimation


def print(): # visualize tree (console)
    """