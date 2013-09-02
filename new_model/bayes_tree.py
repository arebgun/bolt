from __future__ import division
import IPython
import numpy as np
import collections as coll
from utils import shannon_entropy_of_counts, zrm_entropy_of_counts



def count_weighted(iterable, weights, counter=None):
    if counter is None:
        counter = coll.Counter()
    for elem, weight in zip(iterable, weights):
        counter[elem] += weight
    return counter

def weighted_average(numbers, weights):
    return np.dot(numbers, weights)/weights.sum()

def drop_feature(recarray, i):
    tuples = zip(*recarray)
    tuples = tuples[:i] + tuples[i+1:]
    records = zip(*tuples)
    names = recarray.dtype.names[:i] + recarray.dtype.names[i+1:]
    return np.rec.fromrecords(records, names=names)


class BayesianRegressionTree(object):
    """
    """


    def __init__(self, features, labels, weights, max_depth=None, 
                 base_features=None, base_weights=None):
        """ """
        # applicabilities = np.array(labels,dtype=np.float)
        self.node = Node(features, 
                         labels, 
                         weights, 
                         depth=0,
                         max_depth=max_depth,
                         base_features=base_features,
                         base_weights=base_weights)

    def save_for_later():
        pass
        # label_counter = count_weighted(labels, weights)
        # label_set = set(label_counter.keys())
        # label_prior = shannon_entropy_of_counts(label_counter.values())
        # entropies = []
        # for name in features.dtype.names:
        #   data_type = features.dtype.fields[name][0]
        #   if data_type in self.discrete_types:
        #       print name, '(discrete)'
        #       feature_counter = count_weighted(features[name], weights)
        #       mutual = 0
        #       print '  Normal'
        #       for key in feature_counter:
        #           w = np.where(features[name]==key)
        #           lcounter = count_weighted(labels[w],weights[w])
        #           post = shannon_entropy_of_counts(lcounter.values())
        #           gain = label_prior - post
        #           print '   ',name, key
        #           print '      Entropy:', post
        #           print '      InfoGain:', gain
        #           mutual += (feature_counter[key]/
        #                      sum(feature_counter.values()))*gain
        #       print '    Mutual Info', mutual

        #       mutual = 0
        #       feature_prior = shannon_entropy_of_counts(
        #           feature_counter.values())
        #       print '  Reverse'
        #       for label in label_set:
        #           w = np.where(labels==label)
        #           counter = count_weighted(features[w][name], weights[w])
        #           # print "counts", counter
        #           post = shannon_entropy_of_counts(counter.values())
        #           gain = feature_prior - post
        #           print '    positive',label
        #           print '      Entropy:', post
        #           print '      InfoGain:', gain
        #           mutual += (label_counter[label]/
        #                      sum(label_counter.values()))*gain
        #       print '    Mutual Info', mutual

        #       zrm_feature_prior = 
        #       print '  ZRM Reverse'
        #       print
        #   elif data_type in self.continuous_types:
        #       print name, '(continuous)'
        #       hist, bins = np.histogram(features[name], weights=weights)
        #       print hist
        #       entropy = shannon_entropy_of_counts(hist)
        #       entropies.append(entropy)
        #       print 'Shannon entropy:', entropy
        #       print
        #   else:
        #       print data_type
        #       entropies.append(float('inf'))
        # print entropies
        # chosen_i = np.argmin(entropies)
        # print chosen_i
        # chosen_feat = features.dtype.names[chosen_i]
        # print 'Choosing ', chosen_feat



class Node(object):
    """
    """

    discrete_types = [np.bool]
    continuous_types = [np.float]
    def __init__(self, features, labels, weights, depth, max_depth, 
                 base_features=None, base_weights=None, print_space=''):
        """ """
        entropy_of_counts = zrm_entropy_of_counts
        weights = np.copy(weights)

        label_counter = count_weighted(labels, weights)
        label_set = set(label_counter.keys())
        mutuals = []
        infogains = []

        for name in features.dtype.names:
            data_type = features.dtype.fields[name][0]

            mutual = 0
            infogain = -float('inf')
            if data_type in self.discrete_types:
                print print_space, name, '(discrete)'

                N = len(features[name])
                base_feature_counter = count_weighted(base_features[name],
                                                      base_weights)
                base_keys = base_feature_counter.keys()
                base_counts = np.array([base_feature_counter[key] for key in base_keys])

                feature_counter = count_weighted(features[name], weights)
                counts = np.array([feature_counter[key] for key in base_keys])
                # print counts/counts.sum()
                # print base_counts/base_counts.sum()
                # print N
                feature_prior = entropy_of_counts(counts, base_counts, N)
                # print print_space, 'feature prior', feature_prior
                # print print_space, '  Reverse'
                for label in label_set:
                    w = np.where(labels==label)[0]
                    N = len(w)
                    counter = count_weighted(features[w][name], weights[w])
                    # print "counts", counter
                    counts = np.array([counter[key] for key in base_keys])
                    # print counts/counts.sum()
                    # print base_counts/base_counts.sum()
                    # print N
                    post = entropy_of_counts(counts, base_counts, N)
                    gain = feature_prior - post
                    if gain > infogain:
                        infogain = gain
                    print print_space, '   ', label, 'InfoGain:', gain
                    mutual += (label_counter[label]/
                               sum(label_counter.values()))*gain
                mutuals.append(mutual)
                infogains.append(infogain)
                print print_space, '    Mutual Info', mutual

            elif data_type in self.continuous_types:
                mutuals.append(-float('inf'))
                infogains.append(-float('inf'))
                print print_space, name, '(continuous)'
                N = len(features[name])
                base_hist, base_bins = np.histogram(base_features[name], 
                                                    weights=base_weights)
                hist, _ = np.histogram(features[name], bins=base_bins,
                                    weights=weights)
                feature_prior = entropy_of_counts(hist, base_hist, N)

                for label in label_set:
                    w = np.where(labels==label)[0]
                    N = len(w)
                    hist, _ = np.histogram(features[w][name], bins=base_bins,
                                           weights=weights[w])
                    post = entropy_of_counts(hist, base_hist, N)
                    gain = feature_prior - post
                    print print_space, '   ', label, 'InfoGain:', gain
                    mutual += (label_counter[label]/
                               sum(label_counter.values()))*gain
                # mutuals.append(mutual)
                print print_space, '    Mutual Info', mutual
            else:
                raise Exception('Data type: %s' % data_type)

        decider = infogains#mutuals
        chosen_i = np.argmax(decider)
        if decider[chosen_i] < 0.05 or np.isnan(decider[chosen_i]):
            print print_space, 'Quitting branch'
            return

        chosen_name = features.dtype.names[chosen_i]
        print print_space, 'Choosing', chosen_name
        chosen_features = features[chosen_name]
        # print chosen_features[:10]
        feature_counter = count_weighted(chosen_features, weights)
        for key in feature_counter:
            w = np.where(chosen_features==key)
            wavg = weighted_average(labels[w], weights[w])
            print print_space, chosen_name, key, wavg
            new_features = drop_feature(features[w], chosen_i)
            bw = np.where(base_features[chosen_name]==key)
            print print_space, '-'*10
            Node(features=new_features,
                 labels=labels[w],
                 weights=weights[w],
                 depth=depth+1,
                 max_depth=max_depth,
                 base_features=base_features[bw],
                 base_weights=base_weights[bw],
                 print_space=print_space+'    ')

class Leaf(object):
    """
    """