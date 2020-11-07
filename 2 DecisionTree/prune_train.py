
# Hugo Zhang
# This file is for comparing pruning methods.

import numpy as np
import argparse
from get_data import load_data
from structures import C45Tree
from pruning import entropy_with_numnodes, error_with_numnodes, pep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune', required=True, type=str)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--log', default=None, type=str)
    args = parser.parse_args()
    args.prune = args.prune.lower()
    assert args.prune in ('entropy', 'error', 'pep')

    train_feats, train_labels, test_feats, test_labels, attr_dim = load_data()

    if args.prune == 'entropy':
        prune_func = entropy_with_numnodes(args.alpha)
    elif args.prune == 'error':
        prune_func = error_with_numnodes(args.alpha)
    else:
        prune_func, prune_at_last = pep()

    tree = C45Tree(train_feats, train_labels, attr_dim, prune_func)
    if args.prune == 'pep':
        tree = prune_at_last(tree)
    acc = tree.get_acc(test_feats, test_labels)
    log_str = 'c45 tree with prune %s, acc: %s' % (args.prune, acc)
    if args.log is None:
        print(log_str)
    else:
        with open(args.log, 'w') as fw:
            fw.write(log_str + '\n')

