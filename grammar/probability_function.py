import math
import numpy as np
import collections as coll
import itertools as it
import domain as dom
import utils

import scipy.optimize as opt

import IPython

def count_weighted(iterable, weights, counter=None):
    if counter is None:
        counter = coll.Counter()
    for elem, weight in zip(iterable, weights):
        counter[elem] += weight
    return counter

def weighted_gini_purity(instance_labels, instance_weights):
    counts = np.array(
        count_weighted(instance_labels, instance_weights).values())
    return np.sum((counts/counts.sum())**2)

def fraction_correct(torf, assignment, labels, weights):
    w = np.where(assignment==torf)
    s = float(weights[w].sum())
    if s > 0:
        return weights[w][np.where(labels[w]==torf)].sum()/s
    else:
        return 0.0

class ProbabilityFunction(object):
    '''Base class'''

    def __hash__(self):
        raise NotImplementedError

    @staticmethod
    def binomial_error(probs, classes, weights):
        epsilon = 0.000000001
        probs[np.where(probs==0)] += epsilon
        probs[np.where(probs==1)] -= epsilon
        # return np.product((probs**ydata)*((1-probs)**(1-ydata)))
        # return -(classes*np.log(probs) + (1-classes)*np.log(1-probs)).sum()
        return -np.dot(classes*np.log(probs) + (1-classes)*np.log(1-probs), weights)


def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable( it.combinations(xs,n) for n in range(1,len(xs)) )

class DiscreteProbFunc(ProbabilityFunction):

    def __init__(self, pairs):
        # super(DiscreteProbFunc, self).__init__(float)
        # print 'probability_function.py:17: Initing DiscreteProbFunc', pairs
        self.ddict = coll.defaultdict(float)
        self.ddict.update(pairs)

    def __repr__(self):
        return '<DiscretePF %s>' % self.ddict

    def __call__(self, key):
        if isinstance(key, np.ndarray):
            value_set = set(key)
            to_return = np.zeros(len(key))
            for value in value_set:
                to_return[np.where(key==value)] = self.ddict[value]
            return to_return
        else:
            return self.ddict[key]

    def __hash__(self):
        return hash(tuple(self.ddict.items()))

    @staticmethod
    def build_binary(fvalues, real_labels, weights, errfunc):
        result = ''
        value_set = set(fvalues)
        if len(value_set) < 2:
            return None, 'len(value_set) < 2\n'

        scores = []
        for value_subset in powerset(value_set):
            labels = np.in1d(fvalues, value_subset)
            true_negative = fraction_correct(False,labels,real_labels,weights)
            true_positive = fraction_correct(True,labels,real_labels,weights)
            score = (true_negative+true_positive)/2.0
            scores.append((score,value_subset))
        scores.sort(reverse=True)
        # IPython.embed()
        best_set = scores[0][1]
        pairs = zip(best_set,[1.0]*len(best_set))
        to_return = DiscreteProbFunc(pairs)
        return to_return, result


sqrt3 = np.sqrt(3)
class ContinuousProbFunc(ProbabilityFunction):

    def __init__(self, loc, scale, domain):
        self.loc = loc
        self.scale = scale
        self.domain = domain

    def __call__(self, x):
        return self.shape_function(x, self.domain, self.loc, self.scale)

    def __repr__(self):
        return "<%s loc=%s, scale=%s>" % (self.__class__.__name__, 
                                          self.loc, self.scale)

    def __key(self):
        return (self.__class__.__name__, self.loc, self.scale, self.domain)

    def __hash__(self):
        return hash(self.__key())

    def copy(self):
        return self.__class__(self.loc, self.scale, self.domain)

    @staticmethod
    def estimate_parameters(domain, x):
        raise NotImplementedError

    @staticmethod
    def binom_errfunc(params, func, domain, xdata, ydata, weights):
        epsilon = 0.000000001
        probs = func(xdata, domain, *params)
        probs[np.where(probs==0)] += epsilon
        probs[np.where(probs==1)] -= epsilon
        # return np.product((probs**ydata)*((1-probs)**(1-ydata)))
        return -np.dot(ydata*np.log(probs) + (1-ydata)*np.log(1-probs), weights)

    @classmethod
    def fit_function(cls,p0,function,domain,train_xs,train_ys,weights,**kwargs):
        p1 = opt.fmin_l_bfgs_b(func=cls.binom_errfunc,
                               x0=p0,
                               args=(function,domain,train_xs,train_ys,weights),
                               **kwargs)[0]
        return p1

    @classmethod
    def build(cls, domain, xs, ys, weights):
        loc, scale = cls.estimate_parameters(domain, xs, ys, weights)
        if loc is None:
            return None
        xs = np.array(xs,dtype=float)
        ys = np.array(ys,dtype=float)
        kwargs = dict(approx_grad=True, disp=False, maxfun=5)
        loc, scale = cls.fit_function((loc,scale),cls.shape_function,domain,xs,ys,weights,**kwargs)
        return cls(loc, scale, domain)



class CentroidalProbFunc(ContinuousProbFunc):
    '''Functions such that loc and scale are the mean and standard deviation'''
    @staticmethod
    def estimate_parameters(domain, xs, ys, weights):
        w = np.where(ys)
        return domain.sample_mean_and_std(xs[w], weights=weights[w]) #TODO correct for non-uniform distribution (use ys)

class DecayEnvelope(CentroidalProbFunc):
    @staticmethod
    def shape_function(x, domain, loc, scale):
        return np.exp(-math.e*np.abs(domain.norm(x-loc))/
                (2*scale))


class LogisticBell(CentroidalProbFunc):
    @staticmethod
    def shape_function(x, domain, loc, scale):
        if isinstance(x,np.ndarray): x = x.astype(float)
        return np.exp(-np.pi*domain.norm(x-loc)/(scale*sqrt3))\
            /(((1+np.exp(-np.pi*domain.norm(x-loc)/(scale*sqrt3)))/2.)**2)

class GaussianBell(CentroidalProbFunc):
    @staticmethod
    def shape_function(x, domain, loc, scale):
        if isinstance(x,np.ndarray): x = x.astype(float)
        return np.exp(-0.5*(domain.norm(x-loc)/scale)**2)

class SechBell(CentroidalProbFunc):
    '''Mathematically simpler equivalent to logistic bell'''
    @staticmethod
    def shape_function(x, domain, loc, scale):
        if isinstance(x,np.ndarray): x = x.astype(float)
        return 1/(np.cosh((np.pi*domain.norm(x-loc)/
                (2*scale*sqrt3)))**2)

class VonMisesCircularBell(CentroidalProbFunc):
    @staticmethod
    def shape_function(x, domain, loc, scale):
        return np.exp(scale*(np.cos(domain.to_radians(x)-
                                         domain.to_radians(loc))-1))



class SigmoidProbFunc(ContinuousProbFunc):
    @staticmethod
    def estimate_parameters(domain, x, y, weights):#TODO correct for non-uniform distribution (use ys)
        if len(weights) <= 1 or sum(weights) == 0:
            return None, None
        sort_i = np.argsort(x)
        sorted_x = x[sort_i]
        sorted_y = y[sort_i]
        sorted_weights = weights[sort_i]
        diffs = sorted_y[1:]-sorted_y[:-1]
        # diff_x = sorted_x[np.where(diffs)]
        w = np.where(diffs)
        diff_x = (sorted_x[1:][w]+sorted_x[:-1][w])/2.0
        diff_weights = (sorted_weights[1:][w]+sorted_weights[:-1][w])/2.0
        if sum(diff_weights) == 0:
            return None, None
        initial_loc = np.average(diff_x, weights=diff_weights)
        initial_scale = np.average((diff_x-initial_loc)**2, weights=diff_weights)
        w = np.where(sorted_y)
        nw = np.where(np.logical_not(sorted_y))
        if sum(sorted_weights[w]) == 0 or sum(sorted_weights[nw]) == 0:
            return None, None
        mean_diff = np.average(sorted_x[w], weights=sorted_weights[w]) - \
                    np.average(sorted_x[nw], weights=sorted_weights[nw])
        initial_scale = math.copysign(initial_scale,mean_diff)
        return initial_loc, initial_scale*np.sqrt(3)/np.pi

class LogisticSigmoid(SigmoidProbFunc):
    @staticmethod
    def shape_function(x, domain, loc, scale):
        if isinstance(x,np.ndarray): x = x.astype(float)
        return 1./(1+np.exp(-np.pi*(x-loc)/(scale*sqrt3)))

