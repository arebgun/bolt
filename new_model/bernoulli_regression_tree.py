from __future__ import division
import random as rand

import numpy as np
import itertools as it
import collections as coll
import sklearn.linear_model as lm

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

# def weighted_impurity(labels, weights, regress=False, return_counts=False):
#     c = count_weighted(labels, weights)
#     most_common = c.most_common(1)[0][1]
#     all_counts = np.array(c.values()).sum()
#     if return_counts:
#         return (all_counts-most_common)/all_counts, all_counts
#     else:
#         return (all_counts-most_common)/all_counts

def weighted_half_gini_purity(instance_labels, instance_weights):
    pass

def weighted_gini_purity(instance_labels, instance_weights):
    counts = np.array(
        count_weighted(instance_labels, instance_weights).values())
    return np.sum((counts/counts.sum())**2)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def rec_to_float_array(arr):
    return np.array([arr[n] for n in arr.dtype.names], dtype=float).T


class BernoulliRegressionTree(object):

    def __init__(self, feature_array, label_array, weight_array,
                 cost='half_gini', alpha=0, min_likelihood_gain=0.1,
                 min_leaf_instances=10, max_split=np.inf, max_cont=np.inf):

        self.cost = cost
        self.alpha = alpha
        self.min_leaf_instances = min_leaf_instances

        root = DiscreteLeaf(feature_array, label_array, weight_array)

        # Split tree until can't split (returns None), or until max_split
        count = -1
        new_tree = root
        while( new_tree is not None):
            self.tree = new_tree
            count += 1
            if count >= max_split:
                break
            new_tree = self.split(self.tree, cost, alpha, min_leaf_instances)
            # print new_tree

        count = -1
        new_tree = self.tree
        while( new_tree is not None):
            self.tree = new_tree
            count += 1
            if count >= max_cont:
                break
            new_tree = self.add_continuous_node(self.tree)
            # print new_tree

    def __call__(self, instances):
        return self.tree(instances)


    @classmethod
    def split(cls, tree, cost, alpha, min_leaf_instances):
        #TODO: use alpha

        new_tree = tree.copy()
        leaves = new_tree.leaves

        split_features = []
        scores = []
        for leaf in leaves:
            split_feature, split_value, split_score = \
                leaf.select_discrete_feature(cost, min_leaf_instances)
            split_features.append((split_feature, split_value))
            scores.append(split_score)

        split_leaf_i = np.argmin(scores)
        split_score = scores[split_leaf_i]

        if split_score > 1.0:
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
    def add_continuous_node(cls, tree):
        new_tree = tree.copy()
        leaves = new_tree.leaves
        # print leaves

        clfs, scores=zip(*[leaf.select_continuous_feature() for leaf in leaves])
        best_i = np.argmax(scores)

        old_leaf = leaves[best_i]
        new_node = old_leaf.add_continuous_node(clfs[best_i])

        if new_node is None:
            return None
        else:
            return new_tree

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
                alpha_k_tree_scores[i, j]=fold_tree_scores[highest_less_than]*\
                                          weights[test_ind].sum()

        print alpha_k_tree_scores.shape
        mean_scores = alpha_k_tree_scores.sum(axis=0)/weights.sum()
        print mean_scores.shape
        print 'Mean scores:', mean_scores
        min_ind = mean_scores.argmin()
        print 'Best score:',mean_scores[min_ind]
        print 'Best alpha:',alphas[min_ind]

        brt = BernoulliRegressionTree.__new__(BernoulliRegressionTree)
        brt.cost = 'gini'
        brt.alpha = alphas[min_ind]
        brt.min_leaf_instances = min_leaf_instances
        brt.tree = trees[min_ind]
        return brt

    @classmethod
    def cv_init2(cls, instances, labels, weights, 
                cost='gini', min_leaf_instances=10, num_folds=10):

        mlf = min_leaf_instances
        trees = cls.build_greedy_trees(instances, labels, weights, cost=cost,
                                       min_leaf_instances=mlf)
        c1trees = [cls.add_continuous_node(tree) for tree in trees]
        c2trees = [cls.add_continuous_node(tree) for tree in c1trees]

        alphas, _ = cls.find_critical_alphas(trees,instances,labels,weights)
        print 'alphas:',alphas
        alpha_ks = np.sqrt(alphas[:-1]*alphas[1:])#geometric mean of a_k & a_k+1
        print 'alpha_ks:',alpha_ks
        print

        c1alphas, _ = cls.find_critical_alphas(c1trees,instances,labels,weights,
                                               cost='binomial')
        print 'c1alphas:',c1alphas
        c1alpha_ks = np.sqrt(c1alphas[:-1]*c1alphas[1:])
        print 'c1alpha_ks:',c1alpha_ks

        c2alphas, _ = cls.find_critical_alphas(c2trees,instances,labels,weights,
                                               cost='binomial')
        print 'c2alphas:',c2alphas
        c2alpha_ks = np.sqrt(c2alphas[:-1]*c2alphas[1:])
        print 'c2alpha_ks:',c2alpha_ks

        folds_indices = cls.select_folds(len(instances), num_folds)

        alpha_k_tree_scores = np.empty((num_folds, len(alpha_ks)))
        c1alpha_k_tree_scores = np.empty((num_folds, len(c1alpha_ks)))
        c2alpha_k_tree_scores = np.empty((num_folds, len(c2alpha_ks)))
        for i in range(num_folds):
            test_ind = folds_indices[i]
            train_ind = list(it.chain(*(folds_indices[:i]+folds_indices[i+1:])))
            fold_trees = cls.build_greedy_trees(instances[train_ind],
                                                labels[train_ind],
                                                weights[train_ind],
                                                cost=cost,
                                                min_leaf_instances=mlf)

            fold_c1trees = [cls.add_continuous_node(tree) for tree in fold_trees]
            fold_c2trees = [cls.add_continuous_node(tree) for tree in fold_c1trees]

            fold_tree_alphas, fold_tree_scores = \
                cls.find_critical_alphas(fold_trees,
                                         instances[test_ind],
                                         labels[test_ind],
                                         weights[test_ind],
                                         cost='binomial')

            fold_c1_alphas, fold_c1_scores = \
                cls.find_critical_alphas(fold_c1trees,
                                         instances[test_ind],
                                         labels[test_ind],
                                         weights[test_ind],
                                         cost='binomial')

            fold_c2_alphas, fold_c2_scores = \
                cls.find_critical_alphas(fold_c2trees,
                                         instances[test_ind],
                                         labels[test_ind],
                                         weights[test_ind],
                                         cost='binomial')
            print fold_tree_alphas
            print fold_c1_alphas
            print fold_c2_alphas
            print

            for j, alpha_k in enumerate(alpha_ks):
                less_than = fold_tree_alphas < alpha_k
                temp = less_than.nonzero()[0]
                if len(temp) == 0:
                    highest_less_than = -1
                else:
                    highest_less_than = less_than.nonzero()[0][0]
                alpha_k_tree_scores[i, j]=fold_tree_scores[highest_less_than]*\
                                          weights[test_ind].sum()

            for j, alpha_k in enumerate(c1alpha_ks):
                # print alpha_k, fold_c1_alphas
                less_than = fold_c1_alphas < alpha_k
                temp = less_than.nonzero()[0]
                if len(temp) == 0:
                    highest_less_than = -1
                else:
                    highest_less_than = less_than.nonzero()[0][0]
                c1alpha_k_tree_scores[i, j]=fold_c1_scores[highest_less_than]*\
                                          weights[test_ind].sum()

            for j, alpha_k in enumerate(c2alpha_ks):
                # print alpha_k, fold_c2_alphas
                less_than = fold_c2_alphas < alpha_k
                temp = less_than.nonzero()[0]
                if len(temp) == 0:
                    highest_less_than = -1
                else:
                    highest_less_than = less_than.nonzero()[0][0]
                c2alpha_k_tree_scores[i, j]=fold_c2_scores[highest_less_than]*\
                                          weights[test_ind].sum()

        mean_scores = alpha_k_tree_scores.sum(axis=0)/weights.sum()
        print 'Mean scores:', mean_scores
        min_ind = mean_scores.argmin()
        print 'Best score:',mean_scores[min_ind]
        print 'Best likelihood:',1-mean_scores[min_ind]
        print 'Best alpha:',alphas[min_ind]
        print
        c1mean_scores = c1alpha_k_tree_scores.sum(axis=0)/weights.sum()
        print 'C1 Mean scores:', c1mean_scores
        c1min_ind = c1mean_scores.argmin()
        print 'C1 Best score:',c1mean_scores[c1min_ind]
        print 'C1 Best likelihood:',1-c1mean_scores[c1min_ind]
        print 'C1 Best alpha:',c1alphas[c1min_ind]
        print
        c2mean_scores = c2alpha_k_tree_scores.sum(axis=0)/weights.sum()
        print 'C2 Mean scores:', c2mean_scores
        c2min_ind = c2mean_scores.argmin()
        print 'C2 Best score:',c2mean_scores[c2min_ind]
        print 'C2 Best likelihood:',1-c2mean_scores[c2min_ind]
        print 'C2 Best alpha:',c2alphas[c2min_ind]
        print

        brt = BernoulliRegressionTree.__new__(BernoulliRegressionTree)
        brt.cost = 'gini'
        brt.alpha = alphas[min_ind]
        brt.min_leaf_instances = min_leaf_instances
        brt.tree = trees[min_ind]
        return brt

    @classmethod
    def cv_init3(cls, instances, labels, weights, 
                cost='gini', min_leaf_instances=10, max_split=3, max_cont=1,
                num_folds=10):

        mlf = min_leaf_instances
        max_splits = range(max_split+1)
        max_conts = range(max_cont+1)
        options = list(it.product(max_splits,max_conts))

        folds_indices = cls.select_folds(len(instances), num_folds)

        fold_tree_scores = np.empty((num_folds, len(options)))
        for i in range(num_folds):
            test_ind = folds_indices[i]
            train_ind = list(it.chain(*(folds_indices[:i]+folds_indices[i+1:])))

            fold_trees = [BernoulliRegressionTree(instances[train_ind],
                                                 labels[train_ind],
                                                 weights[train_ind],
                                                 cost=cost,
                                                 max_split=max_split,
                                                 max_cont=max_cont,
                                                 min_leaf_instances=mlf)
                          for max_split, max_cont in options]

            fold_tree_scores[i] = [cls.binomial_cost(tree(instances[test_ind]), 
                                                          labels[test_ind], 
                                                          weights[test_ind])
                                   for tree in fold_trees]

        print fold_tree_scores
        mean_scores = fold_tree_scores.sum(axis=0)/weights.sum()
        print mean_scores
        print 'Mean scores:', mean_scores
        min_ind = mean_scores.argmin()
        print 'Best score:',mean_scores[min_ind]
        print 'Best options:',options[min_ind]

        max_split, max_cont = options[min_ind]
        brt = BernoulliRegressionTree(instances,
                                      labels,
                                      weights,
                                      cost=cost,
                                      max_split=max_split,
                                      max_cont=max_cont,
                                      min_leaf_instances=mlf)
        return brt

    @staticmethod
    def binomial_cost(probs, labels, weights):
        notlabels = np.logical_not(labels)
        return 1-np.dot(labels*probs+notlabels*(1-probs), weights)/weights.sum()

    @staticmethod
    def find_critical_alpha(tree, instances, labels, weights, 
                            cost='MC', epsilon=1, return_score=False):
        num_nodes = len(tree.nodes)
        if cost == 'MC':
            counts = count_weighted(labels, weights)
            most_common_label = counts.most_common(1)[0][0]
            root_cost = np.dot(labels!=most_common_label, weights)/weights.sum()
            tree_labels = tree(instances)
            tree_cost = np.dot(labels!=tree_labels, weights)/weights.sum()
        elif cost == 'binomial':
            p = labels.mean()
            l = labels
            notl = np.logical_not(l)
            w = weights

            root_cost = 1-np.dot(labels*p + notl*(1-p), w)/w.sum()
            # root_cost = 1 - 2**root_like
            p = tree(instances)#, regress=True)
            tree_cost = 1-np.dot(labels*p + notl*(1-p), w)/w.sum()
            # tree_cost = 1 - 2**tree_like
            # print root_like, tree_like
            # print root_cost, tree_cost, root_cost-tree_cost

        else:
            raise NotImplementedError('Not implemented for cost %s' % cost)

        # root_cost + alpha == tree_cost + alpha*num_leaves
        # root_cost - tree_cost = alpha*num_leaves - alpha
        # root_cost - tree_cost = alpha*(num_leaves - 1)
        # alpha = (root_cost - tree_cost)/(num_leaves - 1)
        # epsilon to ensure unique values if tree_cost==root_cost repeatedly
        critical_alpha = (epsilon + root_cost - tree_cost)/(num_nodes - 1)
        if return_score:
            return critical_alpha, tree_cost
        else:
            return critical_alpha

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
        self.parent = parent
        self.regress = regress
        self.instances = instances
        self.labels = labels
        self.weights = weights
        self.counts = count_weighted(labels, weights)
        self.most_common = self.counts.most_common(1)[0][0]
        numbers, weights = zip(*self.counts.items())
        self.mean_value = weighted_average(numbers, np.array(weights))

        dtype = instances.dtype
        self.discrete_features = [name for name in dtype.names
                             if dtype.fields[name][0] in self.__discrete_types]
        self.continuous_features = [name for name in dtype.names
                            if dtype.fields[name][0] in self.__continuous_types]
        self.continuous_child = None

    def copy(self, new_parent=None):
        if new_parent is None:
            new_parent = self.parent

        new_leaf = DiscreteLeaf(instances=self.instances,
                            labels=self.labels,
                            weights=self.weights,
                            parent=new_parent,
                            regress=self.regress)
        if self.continuous_child is not None:
            new_leaf.continuous_child = self.continuous_child.copy()
        return new_leaf

    def __call__(self, instances, regress=False):
        to_return = np.empty(len(instances))

        if self.continuous_child is not None:
            return self.continuous_child(instances, regress=regress)

        if regress:
            to_return[:] = self.mean_value
        else:
            to_return[:] = self.most_common
        return to_return

    def best_split(self, feature, cost, min_instances):
        feature_values = list(set(self.instances[feature]))
        scores = []
        #TODO: try all subsets of values?
        for value in feature_values:
            true = self.instances[feature]==value
            false = np.logical_not(true)
            wtrue = np.where(true)[0]
            wfalse = np.where(false)[0]

            if len(wtrue) < min_instances or len(wfalse) < min_instances:
                score = 1.5 #impossible impurity, flag to abort
            else:

                true_score = 1 - weighted_gini_purity(self.labels[wtrue],
                                                      self.weights[wtrue])
                false_score = 1 - weighted_gini_purity(self.labels[wfalse],
                                                       self.weights[wfalse])
                if cost == 'half_gini':
                    score = min(true_score, false_score)
                elif cost == 'two_thirds_gini':
                    minn, maxx = sorted([true_score, false_score])
                    score = (2*minn + maxx)/3
                elif cost == 'gini':
                    score = (true_score + false_score)/2.0
                else:
                    raise NotImplementedError

            scores.append(score)

        best_i = np.argmin(scores)
        return scores[best_i], feature_values[best_i]

    def select_discrete_feature(self, cost, min_instances):

        scores = []
        split_values = []
        for feature in self.discrete_features:
            score, split_value = self.best_split(feature, cost, min_instances)
            scores.append(score)
            split_values.append(split_value)

        # Lower score is always better
        best_feature_i = np.argmin(scores)
        best_score = scores[best_feature_i]
        # new_impurities = feature_impurities[best_feature_i]
        best_feature = self.discrete_features[best_feature_i]
        best_split_value = split_values[best_feature_i]

        return best_feature, best_split_value, best_score

    def select_continuous_feature(self):
        l = self.labels
        notl = np.logical_not(l)
        w = self.weights
        p = self.mean_value

        # original_likelihood = np.dot(l*np.log2(p) + notl*np.log2(1-p), w)
        original_likelihood = np.dot(l*p + notl*(1-p), w)/w.sum()

        likelihood_diffs = []
        for feature in self.continuous_features:

            column = np.array([self.instances[feature]]).T
            clf = ContinuousNode.fit_logistic(column, l, w)

            if clf is not None:
                p = clf.predict_proba(column)
                # likelihood = np.dot(l*np.log2(p) + notl*np.log2(1-p), w)
                likelihood = np.dot(l*p + notl*(1-p),w)/w.sum()

                likelihood_diff = likelihood - original_likelihood
                likelihood_diffs.append(likelihood_diff)
            else:
                likelihood_diffs.append(0.0)
            # print feature, original_likelihood, likelihood, likelihood_diff
        best_i = np.argmin(likelihood_diffs)
        return self.continuous_features[best_i], likelihood_diffs[best_i]
            

    def split(self, feature, operator, value):
        new_node = DiscreteNode(
            decision_feature=feature, 
            decision_operator=operator, 
            decision_value=value,
            instances=self.instances,
            labels=self.labels,
            weights=self.weights,
            parent=self.parent,
            regress=self.regress
            )
        if self.parent is not None:
            if self.parent.true_child == self:
                self.parent.true_child = new_node
            elif self.parent.false_child == self:
                self.parent.false_child = new_node
            else:
                raise Exception('Node has baaad parent\n'+
                                str(self.parent) + '\n'+
                                str(self.parent.true_child) + '\n'+
                                str(self.parent.false_child) + '\n'+
                                str(self) + '\n')
        return new_node

    def add_continuous_node(self, feature):
        if np.all(self.labels) or not np.any(self.labels):
            return None
        else:
            new_node = ContinuousNode(feature,
                                      self.instances,
                                      self.labels,
                                      self.weights)
            self.continuous_child = new_node
            return new_node

    # def __repr__(self):
    #     return 

    @property
    def num_instances(self):
        return len(self.labels)

    @property
    def leaves(self):
        if self.continuous_child is not None:
            return self.continuous_child.leaves
        else:
            return [self]

    @property
    def nodes(self):
        nodes = [self]
        if self.continuous_child is not None:
            nodes.extend(self.continuous_child.nodes)
        return nodes


class DiscreteNode(object):

    discrete_types = [np.bool]
    continuous_types = [np.float]
    def __init__(self, decision_feature, decision_operator, decision_value,
                 instances, labels, weights, parent=None, regress=False):
        self.parent = parent
        self.feature = decision_feature
        self.decision_operator = decision_operator
        self.decision_value = decision_value
        self.instances = instances
        self.labels = labels
        self.weights = weights

        true = decision_operator(instances[decision_feature],decision_value)
        false = np.logical_not(true)
        wtrue = np.where(true)
        wfalse = np.where(false)

        self.true_child = DiscreteLeaf(
            instances=instances[wtrue],
            labels=labels[wtrue],
            weights=weights[wtrue],
            parent=self,
            regress=regress
            )
        self.false_child = DiscreteLeaf(
            instances=instances[wfalse],
            labels=labels[wfalse],
            weights=weights[wfalse],
            parent=self,
            regress=regress
            )

    def copy(self, new_parent=None):
        if new_parent is None:
            new_parent = self.parent
            
        new_node = DiscreteNode.__new__(DiscreteNode)
        new_node.parent = new_parent
        new_node.feature = self.feature
        new_node.decision_operator = self.decision_operator
        new_node.decision_value = self.decision_value
        new_node.true_child = self.true_child.copy(new_parent=new_node)
        new_node.false_child = self.false_child.copy(new_parent=new_node)
        return new_node

    @property
    def leaves(self):
        leaves = []
        for ret in [self.true_child, self.false_child]:
            leaves.extend(ret.leaves)
        return leaves

    @property
    def nodes(self):
        nodes = [self]
        for ret in [self.true_child, self.false_child]:
            nodes.extend(ret.nodes)
        return nodes

    def __call__(self, instances, regress=False):
        results = np.zeros(len(instances))
        t = self.decision_operator(instances[self.feature], self.decision_value)
        w = np.where(t)
        nw = np.where(np.logical_not(t))

        results[w] = self.true_child(instances[w], regress=regress)
        results[nw] = self.false_child(instances[nw], regress=regress)

        return results


class ContinuousNode(object):

    __discrete_types = [np.bool]
    __continuous_types = [np.float]
    def __init__(self, feature, instances, labels, weights):#, function, parameters):
        self.feature = feature
        self.instances = instances
        self.labels = labels
        self.weights = weights

        dtype = instances.dtype
        self.discrete_features = [name for name in dtype.names
                             if dtype.fields[name][0] in self.__discrete_types]
        self.continuous_features = [name for name in dtype.names
                            if dtype.fields[name][0] in self.__continuous_types]

        self.clf = self.fit_logistic(np.array([instances[feature]]).T, 
                                     labels, 
                                     weights)
        self.continuous_child = None

    def __call__(self, instances, regress=False):
        return self.clf.predict_proba(np.array([instances[self.feature]]).T)
        #function(instances[self.feature],**self.parameters)

    def copy(self):
        new_copy = ContinuousNode(self.feature,
                                  self.instances,
                                  self.labels,
                                  self.weights)
        if self.continuous_child is not None:
            new_copy.continuous_child = self.continuous_child.copy()
        return new_copy

    @property
    def leaves(self):
        if self.continuous_child is not None:
            return self.continuous_child.leaves
        else:
            return [self]

    @property
    def nodes(self):
        nodes = [self]
        if self.continuous_child is not None:
            nodes.extend(self.continuous_child.nodes)
        return nodes

    def select_continuous_feature(self):
        l = self.labels
        notl = np.logical_not(l)
        w = self.weights
        p = self(self.instances)

        # original_likelihood = np.dot(l*np.log2(p) + notl*np.log2(1-p), w)
        original_likelihood = np.dot(l*p + notl*(1-p), w)/w.sum()

        likelihood_diffs = []
        for feature in self.continuous_features:

            column = np.array([self.instances[feature]]).T
            clf = ContinuousNode.fit_logistic(column, l, w)

            p = clf.predict_proba(column)
            # likelihood = np.dot(l*np.log2(p) + notl*np.log2(1-p), w)
            likelihood = np.dot(l*p + notl*(1-p),w)/w.sum()

            likelihood_diff = likelihood - original_likelihood
            likelihood_diffs.append(likelihood_diff)
            # print feature, original_likelihood, likelihood, likelihood_diff
        best_i = np.argmin(likelihood_diffs)
        return self.continuous_features[best_i], likelihood_diffs[best_i]

    def add_continuous_node(self, feature):
        if np.all(self.labels) or not np.any(self.labels):
            return None
        else:
            new_node = ContinuousNode(feature,
                                      self.instances,
                                      self.labels,
                                      self.weights)
            self.continuous_child = new_node
            return new_node

    @staticmethod
    def fit_logistic(instances, labels, weights, alpha=0.0001):
        
        if np.all(labels) or not np.any(labels):
            return None
        else:
            sgd = lm.SGDClassifier(loss='log', alpha=0.0001)
            clf = sgd.fit(np.array(instances),labels,sample_weight=weights)
            return clf