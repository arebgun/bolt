from __future__ import division
import random as rand
import numpy as np
import collections as coll
import itertools as it
import IPython

# Utility functions
def count_weighted(iterable, weights, counter=None):
    if counter is None:
        counter = coll.Counter()
    for elem, weight in zip(iterable, weights):
        counter[elem] += weight
    return counter

def weighted_average(numbers, weights):
    return np.dot(numbers, weights)/weights.sum()

def weighted_impurity(labels, weights, regress=False, return_counts=False):
    c = count_weighted(labels, weights)
    most_common = c.most_common(1)[0][1]
    all_counts = np.array(c.values()).sum()
    if return_counts:
        return (all_counts-most_common)/all_counts, all_counts
    else:
        return (all_counts-most_common)/all_counts

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


class BernoulliRegressionTree(object):

    def __init__(self, feature_array, label_array, weight_array,
                 cost='gini', alpha=0, min_leaf_instances=10, 
                 max_splits=np.inf):

        self._cost = 'gini'
        self._alpha = alpha
        self._min_leaf_instances = min_leaf_instances

        root = DiscreteLeaf(feature_array, label_array, weight_array)

        # Split tree until can't split (returns None), or until max_splits
        count = -1
        new_tree = root
        while( new_tree is not None):
            self.tree == new_tree
            count += 1
            if count >= max_splits:
                break
            new_tree = self.split(self.tree, cost, alpha, min_leaf_instances)


    @classmethod
    def split(cls, tree, cost, alpha, min_leaf_instances):
        #TODO: use alpha

        new_tree = tree.copy()
        leaves = new_tree.leaves

        split_features = []
        new_best_impurities = []
        for leaf in leaves:
            split_feature, split_value, split_impurity = \
                leaf.select_feature(min_leaf_instances)
            split_features.append((split_feature, split_value))
            new_best_impurities.append(split_impurity)

        split_leaf_i = np.argmin(new_best_impurities)
        split_impurity = new_best_impurities[split_leaf_i]

        if split_impurity > 0.5:
            return None

        split_feature, split_value = split_features[split_leaf_i]

        old_leaf = leaves[split_leaf_i]
        new_node = old_leaf.split(split_feature, np.equal, split_value)
        if old_leaf == new_tree:
            new_tree = new_node

        return new_tree

        # feature_values = list(set(discrete_feature_array[split_feature]))
        # if len(feature_values) <= 2:
        #     discrete_features.remove(split_feature)
        #     discrete_feature_array = discrete_feature_array[discrete_features]
        # print discrete_features

    @classmethod
    def build_greedy_trees(cls, feature_array, label_array, weight_array,
                 cost='gini', alpha=0, min_leaf_instances=10):

        root = DiscreteLeaf(feature_array, label_array, weight_array)

        # Split tree until can't split (returns None)
        trees = []
        new_tree = root
        while( new_tree is not None):
            trees.append(new_tree)
            # print len(trees)
            new_tree = cls.split(trees[-1], cost, alpha, min_leaf_instances)

        return trees

    @classmethod
    def cv_init(cls, instances, labels, weights, 
                cost='gini', min_leaf_instances=10, num_folds=10):

        mlf = min_leaf_instances
        trees = cls.build_greedy_trees(instances, labels, weights, cost=cost,
                                       min_leaf_instances=mlf)

        alphas, _ = cls.find_critical_alphas(trees,instances,labels,weights)
        alpha_ks = np.sqrt(alphas[:-1]*alphas[1:])#geometric mean of a_k & a_k+1
        print 'alpha_ks:',alpha_ks
        print

        folds_indices = cls.select_folds(len(instances), num_folds)

        alpha_k_tree_scores = np.empty((num_folds, len(alpha_ks)))
        for i in range(num_folds):
            test_ind = folds_indices[i]
            train_ind = list(it.chain(*(folds_indices[:i]+folds_indices[i+1:])))
            fold_trees = cls.build_greedy_trees(instances[train_ind],
                                                labels[train_ind],
                                                weights[train_ind],
                                                cost=cost,
                                                min_leaf_instances=mlf)

            fold_tree_alphas, fold_tree_scores = \
                cls.find_critical_alphas(fold_trees,
                                         instances[test_ind],
                                         labels[test_ind],
                                         weights[test_ind],
                                         cost='MC')

            for j, alpha_k in enumerate(alpha_ks):
                less_than = fold_tree_alphas < alpha_k
                highest_less_than = less_than.nonzero()[0][0]
                alpha_k_tree_scores[i, j] = fold_tree_scores[highest_less_than]

        mean_scores = alpha_k_tree_scores.mean(axis=1)
        print 'Mean scores:', mean_scores
        max_ind = mean_scores.argmax()
        print 'Best score:',mean_scores[max_ind]
        print 'Best alpha:',alphas[max_ind]

        brt = BernoulliRegressionTree.__new__(BernoulliRegressionTree)
        brt._cost = 'gini'
        brt._alpha = alphas[max_ind]
        brt._min_leaf_instances = min_leaf_instances
        brt.tree = trees[max_ind]
        return brt

    def cost_complexity(self, feature_array, label_array):
        raise NotImplementedError

    def num_nodes(self):
        raise NotImplementedError

    def num_leaves(self):
        raise NotImplementedError

    @staticmethod
    def find_critical_alpha(tree, instances, labels, weights, 
                            cost='MC', epsilon=1, return_score=False):
        if cost == 'MC':
            counts = count_weighted(labels, weights)
            most_common_label = counts.most_common(1)[0][0]
            root_cost = np.dot(labels!=most_common_label, weights)/weights.sum()
            tree_labels = tree(instances)
            tree_cost = np.dot(labels!=tree_labels, weights)/weights.sum()
            num_leaves = len(tree.leaves)

            # root_cost + alpha == tree_cost + alpha*num_leaves
            # root_cost - tree_cost = alpha*num_leaves - alpha
            # root_cost - tree_cost = alpha*(num_leaves - 1)
            # alpha = (root_cost - tree_cost)/(num_leaves - 1)

            # epsilon to ensure unique values if tree_cost==root_cost repeatedly
            critical_alpha = (epsilon + root_cost - tree_cost)/(num_leaves - 1)
            if return_score:
                return critical_alpha, tree_cost
            else:
                return critical_alpha
        # elif cost == 'RSS':
        #     pass
        else:
            raise NotImplementedError('Not implemented for cost %s' % cost)

    @classmethod
    def find_critical_alphas(cls, trees, instances, labels, weights, 
                             cost='MC', epsilon=1):
        alphas_scores = [cls.find_critical_alpha(tree=tree,
                                                 instances=instances,
                                                 labels=labels,
                                                 weights=weights,
                                                 cost=cost,
                                                 epsilon=epsilon,
                                                 return_score=True) 
                         for tree in trees]
        alphas, scores = zip(*alphas_scores)
        return np.array(alphas), np.array(scores)

    # @staticmethod
    # def weighted_gini_purity(instance_labels, instance_weights):
    #     counts = np.array(
    #         count_weighted(instance_labels, instance_weights).values())
    #     return np.sum((counts/counts.sum())**2)

    @staticmethod
    def select_folds(num_instances, num_folds):
        per_fold = int(np.ceil(num_instances/num_folds))
        indices = np.arange(num_instances)
        rand.shuffle(indices)
        return list(chunks(indices, per_fold))


class DiscreteLeaf(object):

    __discrete_types = [np.bool]
    __continuous_types = [np.float]
    def __init__(self, instances, labels, weights, parent=None, regress=False):
        self._parent = parent
        self._regress = regress
        self._instances = instances
        self._labels = labels
        self._weights = weights
        self._counts = count_weighted(labels, weights)
        self._most_common = self._counts.most_common(1)[0][0]
        numbers, weights = zip(*self._counts.items())
        self._mean_value = weighted_average(numbers, np.array(weights))

        dtype = instances.dtype
        self._discrete_features = [name for name in dtype.names
                             if dtype.fields[name][0] in self.__discrete_types]

    def copy(self, new_parent=None):
        if new_parent is None:
            new_parent = self._parent

        return DiscreteLeaf(instances=self._instances,
                            labels=self._labels,
                            weights=self._weights,
                            parent=new_parent,
                            regress=self._regress)

    def __call__(self, instances):
        to_return = np.empty(len(instances))
        if self._regress:
            to_return[:] = self._mean_value
        else:
            to_return[:] = self._most_common
        return to_return

    def select_feature(self, min_instances):

        best_impurities = []
        best_values = []
        for feature in self._discrete_features:
            feature_values = list(set(self._instances[feature]))

            impurities = []
            #TODO: try all subsets of values?
            for value in feature_values:
                true = self._instances[feature]==value
                false = np.logical_not(true)
                wtrue = np.where(true)[0]
                wfalse = np.where(false)[0]
                if len(wtrue) < min_instances or len(wfalse) < min_instances:
                    impurity = 1.0 #impossible impurity, flag to abort
                else:
                    impurity = weighted_impurity(self._labels[wtrue],
                                                 self._weights[wtrue])
                impurities.append(impurity)
            # feature_impurities.append(impurities)
            best_i = np.argmin(impurities)
            best_impurities.append(impurities[best_i])
            best_values.append(feature_values[best_i])

        best_feature_i = np.argmin(best_impurities)
        best_impurity = best_impurities[best_feature_i]
        # new_impurities = feature_impurities[best_feature_i]
        best_feature = self._discrete_features[best_feature_i]
        best_value = best_values[best_feature_i]

        return best_feature, best_value, best_impurity

    def split(self, feature, operator, value):
        new_node = DiscreteNode(
            decision_feature=feature, 
            decision_operator=operator, 
            decision_value=value,
            instances=self._instances,
            labels=self._labels,
            weights=self._weights,
            parent=self._parent,
            regress=self._regress
            )
        if self._parent is not None:
            if self._parent.true_return == self:
                self._parent.true_return = new_node
            elif self._parent.false_return == self:
                self._parent.false_return = new_node
            else:
                raise Exception('Node has baaad parent\n'+
                                str(self._parent) + '\n'+
                                str(self._parent.true_return) + '\n'+
                                str(self._parent.false_return) + '\n'+
                                str(self) + '\n')
        return new_node

    # def __repr__(self):
    #     return 

    @property
    def num_instances(self):
        return len(self._labels)

    @property
    def leaves(self):
        return [self]


class Node(object):
    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, feature_name):
        self._feature = feature_name

class DiscreteNode(Node):

    discrete_types = [np.bool]
    continuous_types = [np.float]
    def __init__(self, decision_feature, decision_operator, decision_value,
                 instances, labels, weights, parent=None, regress=False):
        self._parent = parent
        self.feature = decision_feature
        self.decision_operator = decision_operator
        self.decision_value = decision_value
        # self.true_return = true_return
        # self.false_return = false_return

        # IPython.embed()
        true = decision_operator(instances[decision_feature],decision_value)
        false = np.logical_not(true)
        wtrue = np.where(true)
        wfalse = np.where(false)

        self.true_return = DiscreteLeaf(
            instances=instances[wtrue],
            labels=labels[wtrue],
            weights=weights[wtrue],
            parent=self,
            regress=regress
            )
        self.false_return = DiscreteLeaf(
            instances=instances[wfalse],
            labels=labels[wfalse],
            weights=weights[wfalse],
            parent=self,
            regress=regress
            )

    def copy(self, new_parent=None):
        if new_parent is None:
            new_parent = self._parent
            
        new_node = DiscreteNode.__new__(DiscreteNode)
        new_node._parent = new_parent
        new_node.feature = self.feature
        new_node.decision_operator = self.decision_operator
        new_node.decision_value = self.decision_value
        new_node.true_return = self.true_return.copy(new_parent=new_node)
        new_node.false_return = self.false_return.copy(new_parent=new_node)
        return new_node

    @property
    def leaves(self):
        leaves = []
        for ret in [self.true_return, self.false_return]:
            leaves.extend(ret.leaves)
        return leaves

    @classmethod
    def __cv_init(class_, feature_array, label_array, weight_array, 
                  num_folds=10):
        folds_indices = class_.select_folds(len(feature_array),num_folds)

        for fold_indices in folds_indices:
            pass

        pass



    def __call__(self, instances):
        results = np.zeros(len(instances))
        t = self.decision_operator(instances[self.feature], self.decision_value)
        w = np.where(t)
        nw = np.where(np.logical_not(t))

        # if callable(self.true_return):
        results[w] = self.true_return(instances[w])
        # else:
        #     results[w] = self.true_return

        # if callable(self.false_return):
        results[nw] = self.false_return(instances[nw])
        # else:
        #     results[nw] = self.false_return

        return results


class ContinuousNode(Node):

    def __init__(self, feature, function, parameters):
        self.feature = feature
        self.function = function
        self.parameters = parameters

    def __call__(self, instances):
        return self.function(instances[self.feature],**self.parameters)