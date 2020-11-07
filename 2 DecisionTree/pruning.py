
# Hugo Zhang
# This file contains pruning methods.

import numpy as np

def entropy_with_numnodes(alpha):
    # method taught in class
    def prune_func(node, labels, tree):
        cur_entros = len(labels) * tree.get_info_gain(labels)
        if isinstance(node, tree.LeafNode):
            node.entros = cur_entros
            node.leaves = 1
        else:
            son_entros = sum([s.entros for s in node.sons])
            son_leaves = sum([s.leaves for s in node.sons])
            if cur_entros + alpha > son_entros + alpha * son_leaves:
                node.entros = son_entros
                node.leaves = son_leaves
            else:
                node = tree.LeafNode(tree.get_max_label(labels))
                node.entros = cur_entros
                node.leaves = 1
        return node
    return prune_func

def error_with_numnodes(alpha):
    # use error instead of entropy to indicate loss
    # may perform better if using a separate validate set
    def prune_func(node, labels, tree):        
        cur_lab = tree.get_max_label(labels)
        cur_error = np.sum(labels != cur_lab)
        if isinstance(node, tree.LeafNode):
            node.err = cur_error
            node.leaves = 1
        else:
            son_error = sum([s.err for s in node.sons])
            son_leaves = sum([s.leaves for s in node.sons])
            if cur_error + alpha > son_error + alpha * son_leaves:
                node.err = son_error
                node.leaves = son_leaves
            else:
                node = tree.LeafNode(tree.get_max_label(labels))
                node.err = cur_error
                node.leaves = 1
        return node
    return prune_func

def pep():
    # pessimistic error pruning
    # this method is from top to bottom, while the former two is from bottom to top
    # record information for each nodes during tree building; perform pruning at last
    def add_attr(node, labels, tree):
        if isinstance(node, tree.LeafNode):
            node.err = np.sum(labels != node.label)
            node.leaves = 1
        else:
            node.err = sum([s.err for s in node.sons])
            node.leaves = sum([s.leaves for s in node.sons])
        node.labels = labels
        node.sample_num = len(labels)
        return node

    def prune_at_last(tree):
        def prune_sons(node):
            sons = node.sons
            for i in range(len(sons)):
                s = sons[i]
                eerr = s.err + 0.5 * s.leaves
                varerr = np.sqrt(eerr * (1 - eerr / s.sample_num))
                new_lab = tree.get_max_label(s.labels)
                after_err = np.sum(s.labels != new_lab) + 0.5
                if eerr + varerr > after_err:
                    sons[i] = tree.LeafNode(new_lab)
                else:
                    prune_sons(s)
        root_node = tree.root_node
        eerr = root_node.err + 0.5 * root_node.leaves
        varerr = np.sqrt(eerr * (1 - eerr / root_node.sample_num))
        after_err = np.sum(root_node.labels != tree.get_max_label(root_node.labels)) + 0.5
        if eerr + varerr > after_err:
            root_node = tree.LeafNode(root_node.labels)
        else:
            prune_sons(root_node)
        tree.root_node = root_node
        return tree

    return add_attr, prune_at_last
