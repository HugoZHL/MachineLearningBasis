
# Hugo Zhang
# This file is for comparing trees.

import numpy as np
import argparse
from get_data import load_data
from structures import C45Tree, CartTree

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', required=True, type=str)
    parser.add_argument('--log', default=None, type=str)
    args = parser.parse_args()
    args.tree = args.tree.lower()
    assert args.tree in ('c45', 'cart')

    train_feats, train_labels, test_feats, test_labels, attr_dim = load_data()

    if args.tree == 'c45':
        tree = C45Tree(train_feats, train_labels, attr_dim)
    else:
        tree = CartTree(train_feats, train_labels, attr_dim)
    
    acc = tree.get_acc(test_feats, test_labels)
    log_str = '%s tree, acc: %s' % (args.tree, acc)
    if args.log is None:
        print(log_str)
    else:
        with open(args.log, 'w') as fw:
            fw.write(log_str + '\n')

