# Hugo Zhang
# This file is for comparing RandomForest and Bagging classifiers.

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.datasets import *
import numpy as np


def runRandomForest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    return clf.score(test_x, test_y)


def runBagging(train_x, train_y, test_x, test_y):
    clf = BaggingClassifier()
    clf.fit(train_x, train_y)
    return clf.score(test_x, test_y)


def coreExperiment(data, target):
    # print(len(data))
    print('Data:', data.shape)
    print('Target:', target.shape)
    num_samples = len(data)
    train_part = int(num_samples * 0.8)
    rf_acc = []
    bg_acc = []
    for _ in range(5):
        perm = np.random.permutation(len(data))
        data = data[perm]
        target = target[perm]
        train_x = data[:train_part]
        train_y = target[:train_part]
        test_x = data[train_part:]
        test_y = target[train_part:]
        rf_acc.append(runRandomForest(train_x, train_y, test_x, test_y))
        bg_acc.append(runBagging(train_x, train_y, test_x, test_y))
    
    print('Random forest test accuracy: %f.' % (sum(rf_acc) / 5))
    print('Bagging test accuracy: %f.' % (sum(bg_acc) / 5))


def toyExperiment(data, target, name):
    print('%s dataset:' % name)
    coreExperiment(data, target)
    print()


def realExperiment(obj, name):
    print('%s dataset:' % name)
    coreExperiment(obj.data, obj.target)
    print()


toyExperiment(*load_iris(return_X_y=True), 'Iris')
toyExperiment(*load_digits(return_X_y=True), 'Digits')
toyExperiment(*load_wine(return_X_y=True), 'Wine')
toyExperiment(*load_breast_cancer(return_X_y=True), 'Breast cancer')

realExperiment(fetch_olivetti_faces(), 'Olivetti faces')
