import math
import numpy as np
import collections as coll
import domain as dom



class ProbabilityFunction(object):
    '''Base class'''

    def __hash__(self):
        raise NotImplementedError

    @staticmethod
    def binomial_error(probs, classes):
        epsilon = 0.000000001
        probs[np.where(probs==0)] += epsilon
        probs[np.where(probs==1)] -= epsilon
        # return np.product((probs**ydata)*((1-probs)**(1-ydata)))
        return -(classes*np.log(probs) + (1-classes)*np.log(1-probs)).sum()

class DiscreteProbFunc(ProbabilityFunction):

    def __init__(self, pairs):
        # super(DiscreteProbFunc, self).__init__(float)
        # print 'probability_function.py:17: Initing DiscreteProbFunc', pairs
        self.ddict = coll.defaultdict(float)
        self.ddict.update(pairs)

    def __repr__(self):
        return '<DiscretePF %s>' % self.ddict

    def __call__(self, key):
        return self.ddict[key]

    def __hash__(self):
        return hash(tuple(self.ddict.items()))

class ContinuousProbFunc(ProbabilityFunction):
    sqrt3 = np.sqrt(3)

    def __init__(self, loc, scale, domain):
        self.loc = loc
        self.scale = scale
        self.domain = domain

    def __call__(self, x):
        raise NotImplementedError

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



class CentroidalProbFunc(ContinuousProbFunc):
    '''Functions such that loc and scale are the mean and standard deviation'''
    @staticmethod
    def estimate_parameters(domain, xs, ys):
        return domain.sample_mean_and_std(xs[np.where(ys)]) #TODO correct for non-uniform distribution (use ys)


class DecayEnvelope(CentroidalProbFunc):
    def __call__(self, x):
        return np.exp(-math.e*np.abs(self.domain.norm(x-self.loc))/
                (2*self.scale))

class LogisticBell(CentroidalProbFunc):
    def __call__(self, x):
        return np.exp(-np.pi*self.domain.norm(x-self.loc)/(self.scale*self.sqrt3))\
            /(((1+np.exp(-np.pi*self.domain.norm(x-self.loc)/(self.scale*self.sqrt3)))/2.)**2)

class GaussianBell(CentroidalProbFunc):
    def __call__(self, x):
        return np.exp(-0.5*(self.domain.norm(x-self.loc)/self.scale)**2)

class SechBell(CentroidalProbFunc):
    '''Mathematically simpler equivalent to logistic bell'''
    def __call__(self, x):
        return 1/(np.cosh((np.pi*self.domain.norm(x-self.loc)/
                (2*self.scale*self.sqrt3)))**2)

class VonMisesCircularBell(CentroidalProbFunc):
    def __call__(self, x):
        return np.exp(self.scale*(np.cos(self.domain.to_radians(x)-
                                         self.domain.to_radians(self.loc))-1))



class SigmoidProbFunc(ContinuousProbFunc):
    @staticmethod
    def estimate_parameters(domain, x, y):#TODO correct for non-uniform distribution (use ys)
        sort_i = np.argsort(x)
        sorted_x = x[sort_i]
        sorted_y = y[sort_i]
        diffs = sorted_y[1:]-sorted_y[:-1]
        # diff_x = sorted_x[np.where(diffs)]
        diff_x = (sorted_x[np.where(diffs)]+
                 sorted_x[np.where(diffs[1:])])/2.0
        initial_loc = diff_x.mean()
        initial_scale = diff_x.std()
        mean_diff = sorted_x[np.where(sorted_y)].mean() - \
                    sorted_x[np.where(np.logical_not(sorted_y))].mean()
        initial_scale = math.copysign(initial_scale,mean_diff)
        return initial_loc, initial_scale*np.sqrt(3)/np.pi

class LogisticSigmoid(SigmoidProbFunc):
    sqrt3 = np.sqrt(3)
    def __call__(self, x):
        return 1./(1+np.exp(-np.pi*(x-self.loc)/(self.scale*self.sqrt3)))

