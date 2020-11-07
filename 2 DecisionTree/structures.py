
# Hugo Zhang
# This file is for building trees.

import numpy as np

class C45Tree(object):
    class TreeNode(object):
        def __init__(self, attr_index):
            self.attr_index = attr_index
        
        def set_sons(self, sons):
            self.sons = sons

    class LeafNode(object):
        def __init__(self, label):
            self.label = label

    def get_entropy(self, lens):
        entros = 0.0
        all_len = sum(lens)
        for i in lens:
            if i == 0:
                continue
            cur_div = i / all_len
            entros -= cur_div * np.log2(cur_div)
        return entros

    def get_info_gain(self, labels):
        catos = [np.sum([labels == i]) for i in range(self.max_dim)]
        return self.get_entropy(catos)

    def get_max_label(self, labels):
        catos = [np.sum([labels == i]) for i in range(self.max_dim)]
        return np.argmax(catos)

    def get_acc(self, test_feats, test_labels):
        corr = 0
        for feat, lab in zip(test_feats, test_labels):
            node = self.root_node
            while not isinstance(node, self.LeafNode):
                node = node.sons[feat[node.attr_index]]
            corr += (node.label == lab)
        return corr / len(test_labels)

    def build_tree(self, feats, labels, visited):
        if all(visited):
            return self.prune(self.LeafNode(self.get_max_label(labels)), labels, self)
        max_info_gain = 0
        max_ind = -1
        for i in range(self.attr_num):
            if visited[i]:
                continue
            denom = self.get_entropy([np.sum([feats[:, i] == j]) for j in range(self.attr_dim[i])])
            if (denom == 0):
                return self.prune(self.LeafNode(self.get_max_label(labels)), labels, self)
            numer = self.get_info_gain(labels)
            for j in range(self.attr_dim[i]):
                boolean = feats[:, i] == j
                temp_num = np.sum(boolean)
                numer -= temp_num / len(labels) * self.get_info_gain(labels[boolean])
            info_gain = numer / denom
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_ind = i
        
        if max_ind == -1:
            return self.prune(self.LeafNode(self.get_max_label(labels)), labels, self)
        
        visited[max_ind] = True
        node = self.TreeNode(max_ind)
        sons = []
        for i in range(self.attr_dim[max_ind]):
            tmpfeats = feats[feats[:, max_ind] == i]
            tmplabels = labels[feats[:, max_ind] == i]
            sons.append(self.build_tree(tmpfeats, tmplabels, visited))
        node.set_sons(sons)
        visited[max_ind] = False
        return self.prune(node, labels, self)
    
    def __init__(self, train_feats, train_labels, attr_dim, prune_func=None):
        self.attr_dim = attr_dim
        self.max_dim = max(attr_dim)
        self.attr_num = train_feats.shape[-1]
        self.prune = prune_func if prune_func else lambda n, l, s: n
        self.root_node = self.build_tree(train_feats, train_labels, [False for i in range(self.attr_num)])


class CartTree(object):    
    class TreeNode(object):
        def __init__(self, attr_index, attr_judge):
            self.attr_index = attr_index
            self.attr_judge = attr_judge
        
        def set_sons(self, left, right):
            self.left = left
            self.right = right

    class LeafNode(object):
        def __init__(self, label):
            self.label = label

    def get_gini(self, labels):
        ans = 1.0
        all_num = len(labels)
        if all_num == 0:
            return 0
        for i in range(self.max_dim):
            ans -= (np.sum(labels == i) / all_num) ** 2
        return ans

    def get_max_label(self, labels):
        catos = [np.sum([labels == i]) for i in range(self.max_dim)]
        return np.argmax(catos)

    def get_acc(self, test_feats, test_labels):
        corr = 0
        for feat, lab in zip(test_feats, test_labels):
            node = self.root_node
            while not isinstance(node, self.LeafNode):
                if feat[node.attr_index] == node.attr_judge:
                    node = node.left
                else:
                    node = node.right
            corr += (node.label == lab)
        return corr / len(test_labels)

    def build_tree(self, feats, labels, depth):
        if len(np.unique(labels)) == 1 or depth == 15:
            return self.LeafNode(self.get_max_label(labels))
        
        min_gini = 10000000
        min_ind = -1
        min_judge = -1
        for i in range(self.attr_num):
            for j in range(self.attr_dim[i]):
                if (i, j) in self.visited:
                    continue
                left_num = np.sum(feats[:, i] == j)
                right_num = np.sum(feats[:, i] != j)
                all_num = len(labels)
                assert left_num + right_num == all_num
                tmp_gini = left_num / all_num * self.get_gini(labels[feats[:, i] == j]) + \
                    right_num / all_num * self.get_gini(labels[feats[:, i] != j])
                if tmp_gini < min_gini:
                    min_gini = tmp_gini
                    min_ind = i
                    min_judge = j
        
        if min_ind == -1:
            # for threshold if any
            return self.LeafNode(self.get_max_label(labels))
        self.visited.add((min_ind, min_judge))
        node = self.TreeNode(min_ind, min_judge)
        left_mask = feats[:, min_ind] == min_judge
        left_son = self.build_tree(feats[left_mask], labels[left_mask], depth+1)
        right_mask = feats[:, min_ind] != min_judge
        right_son = self.build_tree(feats[right_mask], labels[right_mask], depth+1)
        node.set_sons(left_son, right_son)
        self.visited.remove((min_ind, min_judge))
        return node
    
    def __init__(self, train_feats, train_labels, attr_dim):
        self.attr_dim = attr_dim
        self.attr_num = train_feats.shape[-1]
        self.max_dim = max(attr_dim)
        self.visited = set()
        self.root_node = self.build_tree(train_feats, train_labels, 0)
