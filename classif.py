# -*- coding: utf-8 -*-
"""
@author: alexandrebarachant
"""
import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class AddMeta(BaseEstimator, TransformerMixin):

    def __init__(self, meta=None):
        self.meta = meta

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        if self.meta is not None:
            return numpy.c_[X, self.meta]
        else:
            return X

    def fit_transform(self, X, y=None):
        return self.transform(X)


###############################################################################

def updateMeta(clf, Meta):
    if 'addmeta' in clf.named_steps:
        clf.set_params(addmeta__meta=Meta)


def baggingIterator(opts, users):
    mdls = opts['bagging']['models']
    bag_size = 1 - opts['bagging']['bag_size']
    bag_size = int(numpy.floor(bag_size * len(users)))
    if bag_size == 0:
        return [[u] for u in users]
    else:
        return [numpy.random.choice(users, size=bag_size, replace=False) for i in range(mdls)]
