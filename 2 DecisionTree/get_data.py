
# Hugo Zhang
# This file is for downloading and preprocessing data.
# Here we choose a categorical data from UCI:
# http://archive.ics.uci.edu/ml/datasets/Car+Evaluation


import os
import numpy as np
import urllib.request


def load_data():
    # read in data
    feats = np.load('feats.npy')
    labels = np.load('labels.npy')
    num_samples = feats.shape[0]
    num_attrs = feats.shape[1]
    print('Feature shape: ', feats.shape)
    print('Label shape: ', labels.shape)

    # shuffling
    seq = np.arange(num_samples)
    np.random.shuffle(seq)
    # print(seq[:10])
    feats = feats[seq]
    labels = labels[seq]

    # split data
    num_train = int(0.8 * num_samples)
    num_test = num_samples - num_train
    train_feats = feats[:num_train]
    train_labels = labels[:num_train]
    test_feats = feats[num_train:]
    test_labels = labels[num_train:]
    print('Feature shape for training and testing: ', train_feats.shape, train_labels.shape)
    print('Label shape for training and testing: ', test_feats.shape, test_labels.shape)
    # ensure train and test have all labels
    for i in range(4):
        xtrain = len(train_labels[train_labels == i])
        xtest = len(test_labels[test_labels == i])
        print('Training ratio for attribute %d: %f' % (i, xtrain / (xtrain + xtest)))

    attr_dim = [4, 4, 4, 3, 3, 3]
    return train_feats, train_labels, test_feats, test_labels, attr_dim


if __name__ == '__main__':
    attrs = []
    labs = None
    cate_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.c45-names'
    data_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    name_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names'

    with urllib.request.urlopen(cate_path) as fr:
        for line in fr:
            line = line.decode('utf-8')
            if line[0] == '|' or line[0] == '\n':
                continue
            if labs is None:
                labs = [en.strip(' \n') for en in line.split(',')]
            else:
                line = line.split(':')[1]
                attrs.append([en.strip(' .\n') for en in line.split(',')])

    for i, attr in enumerate(attrs):
        attrs[i] = {k: v for v, k in enumerate(attr)}
    labs = {k: v for v, k in enumerate(labs)}

    print('Attributes: ', attrs)
    print('Labels: ', labs)
        
    Xs = []
    ys = []
    with urllib.request.urlopen(data_path) as fr:
        for line in fr:
            items = line.decode('utf-8').rstrip('\n').split(',')
            assert len(items) == 7
            curX = [attrs[i][items[i]] for i in range(6)]
            Xs.append(curX)
            ys.append(labs[items[-1]])

    Xs = np.array(Xs)
    ys = np.array(ys)
    print("Xs' shape: ", Xs.shape)
    print("ys' shape: ", ys.shape)

    np.save('feats.npy', Xs)
    np.save('labels.npy', ys)
