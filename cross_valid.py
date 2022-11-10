# -*- coding: utf-8 -*-

import yaml
import sys
import numpy as np
import pandas as pd
from time import time

from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from collections import OrderedDict
from multiprocessing import Pool
from functools import partial

# local import
from classif import updateMeta, baggingIterator


def from_yaml_to_func(method, params):
    prm = dict()
    if params is not None:
        for key, val in params.items():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


def make_cv(kfolds, X, Labels, User, Meta, clf, opts):
    users = np.unique(User)
    toPredData = []
    Gauc = []
    for train_users, test_users in kfolds[1]:
        allProb = 0
        test_index = np.array([True if u in set(users[test_users]) else False for u in User])

        if 'bagging' in opts:
            bagging = baggingIterator(opts, [users[i] for i in train_users])
        else:
            bagging = [[-1]]

        for bag in bagging:
            bagUsers = np.array([True if u in set(bag) else False for u in User])
            train_index = np.logical_xor(np.logical_not(test_index), bagUsers)

            try:
                # train
                updateMeta(clf, Meta[train_index])
                clf.fit(X[train_index, :, :], Labels[train_index])

                # predict
                prob = []
                for ut in np.unique(users[test_users]):
                    updateMeta(clf, Meta[User == ut, ...])
                    prob.extend(clf.predict(X[User == ut, ...]))
                prob = np.array(prob)

                allProb += prob / len(bagging)
            except:
                print(kfolds[0])
                print([users[i] for i in train_users])
                print(bag)
                continue

        # save & return
        predictions = OrderedDict()
        predictions['user'] = User[test_index]
        predictions['label'] = Labels[test_index]
        predictions['prediction'] = allProb
        if 'leak' in opts:
            predictions['prediction'] += opts['leak']['coeff'] * (1 - Meta[test_index, -1])
        predictions = pd.DataFrame(predictions)

        Gauc.append(roc_auc_score(predictions.label, predictions.prediction))
        toPredData.append(predictions)
    predData = pd.concat(toPredData)

    Sauc = [roc_auc_score(predData.loc[predData.user == i].label, predData.loc[predData.user == i].prediction) for i in
            np.unique(predData.user)]

    print('Rep %d: gAUC (mean of folds) %0.5f, sAUC %0.5f (%0.5f)' % (
        kfolds[0], np.mean(Gauc), np.mean(Sauc), np.std(Sauc)))

    return [Gauc, Sauc]


# load parameters file
yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

# imports 
for pkg, functions in yml['imports'].items():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['pipeline']:
    for method, params in item.items():
        pipe.append(from_yaml_to_func(method, params))

# create pipeline
clf = make_pipeline(*pipe)
opts = yml['MetaPipeline']
if opts is None:
    opts = {}
# load files
X = np.load('./preproc/epochs.npy')
Labels, User = np.load('./preproc/infos.npy')
Meta = np.load('./preproc/meta_leak.npy') if 'leak' in opts else np.load('./preproc/meta.npy')
users = np.unique(User)

# parallel CV
np.random.seed(5)
folds = yml['CrossVal']['folds']
repetitions = yml['CrossVal']['repetitions']
cores = yml['CrossVal']['cores']

kfolds = [[i, list(KFold(folds, shuffle=True).split(users))] for i in range(repetitions)]
np.random.seed(432432)
t = time()
pMakeCV = partial(make_cv, X=X, Labels=Labels, User=User, Meta=Meta, clf=clf,
                  opts=opts)  # pool function is able to process only 1 argument, so the rest has to be set fixed
pool = Pool(processes=cores)  # define number of cores
results = pool.map(pMakeCV, kfolds, chunksize=1)  # apply parallel processing
pool.close()  # close parallel processes after execution (frees memory)
print("Done in " + str(time() - t) + " second")
# calculating performance

gAUC = np.concatenate([i[0] for i in results])  # mean of folds
sAUC = [np.mean(i[1]) for i in results]
indAUC = np.array([i[1] for i in results])
indAUC = np.mean(indAUC, axis=0)

print('Global AUC : %.5f (%.5f)' % (np.mean(gAUC), np.std(gAUC)))
print('Subject AUC : %.5f (%.5f)' % (np.mean(sAUC), np.std(sAUC)))

# writing it down
import os

comment = yml['CrossVal']['comments']
path = yml['CrossVal']['path']
pipelineSteps = [str(clf.steps[i][1]).replace('\n', '').replace(' ', '') for i in range(len(clf.steps))]
if not os.path.isfile(path):
    fd = open(path, 'w')
    fd.write('comment;folds;reps;gAUC mean;gAUC std;sAUC mean;sAUC std;user' + ";user".join(
        map(str, list(map(int, users)))) + ';leak;bagging;pipeline\n')
    fd.close()
fd = open(path, 'a')
leakStr = 'on' if 'leak' in opts else 'off'
bagStr = '-'.join([str(opts['bagging']['bag_size']), str(opts['bagging']['models'])]) if 'bagging' in opts else 'off'
toWrite = [comment] + list(map(str, [folds, repetitions, np.mean(gAUC), np.std(gAUC), np.mean(sAUC), np.std(sAUC)])) + [
    str(i) for i in indAUC] + [leakStr, bagStr] + pipelineSteps
fd.write(';'.join(toWrite) + '\n')
fd.close()
