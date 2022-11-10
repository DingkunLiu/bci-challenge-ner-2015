# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:19:34 2014

@author: alexandre barachant
"""
import numpy as np
import glob
import re
from scipy import signal
import pandas as pd


def bandpass(sig, band, fs):
    B, A = signal.butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return signal.lfilter(B, A, sig, axis=0)


for test in [False, True]:

    prefix = '' if test is False else 'test_'
    DataFolder = '../data/train/' if test is False else '../data/test/'
    list_of_files = glob.glob(DataFolder + 'Data_*.csv')
    list_of_files.sort()

    reg = re.compile('\d+')

    freq = 200.0

    epoc_window = int(1.3 * freq)

    X = []
    User = []
    idFeedBack = []
    Session = []
    Feedback = []
    Letter = []
    Word = []
    FeedbackTot = []
    LetterTot = []
    WordTot = []

    for f in list_of_files:
        print(f)
        user, session = reg.findall(f)
        sig = np.array(pd.io.parsers.read_csv(f))

        EEG = sig[:, 1:-2]
        EOG = sig[:, -2]
        Trigger = sig[:, -1]

        sigF = bandpass(EEG, [1.0, 40.0], freq)

        idxFeedBack = np.where(Trigger == 1)[0]

        for fbkNum, idx in enumerate(idxFeedBack):
            X.append(sigF[idx:idx + epoc_window, :])
            User.append(int(user))
            idFeedBack.append('S' + user + '_Sess' + session + '_FB' + '%03d' % (fbkNum + 1))
            Session.append(int(session))
            Feedback.append(fbkNum)
            Letter.append(np.mod(fbkNum, 5) + 1)
            Word.append(np.floor(fbkNum / 5) + 1)
            FeedbackTot.append(fbkNum + (int(session) - 1) * 60)
            WordTot.append(np.floor(fbkNum / 5) + 1 + (int(session) - 1) * 12)

    Meta = np.array([Session, Feedback, Letter, Word, FeedbackTot, WordTot]).transpose()

    Meta2 = pd.read_csv('metadata.csv')
    currentUserSet = [True if val in set(User) else False for val in Meta2.subject]

    isLong = Meta2.isLong[currentUserSet].values
    Meta = np.c_[Meta, isLong]

    onlineErr = Meta2.onlineErr[currentUserSet].values
    longProp = Meta2.longProp[currentUserSet].values
    Meta_Leak = np.c_[Meta, onlineErr, longProp]

    if test is False:
        Labels = np.genfromtxt(DataFolder + 'TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]
        info = np.array([Labels, User])
    else:
        info = np.array([idFeedBack, User])

    X = np.array(X).transpose((0, 2, 1))  # Ns,Ne,Nt

    np.save(prefix + 'infos.npy', info)
    np.save(prefix + 'epochs.npy', X)
    np.save(prefix + 'meta.npy', Meta)
    np.save(prefix + 'meta_leak.npy', Meta_Leak)
