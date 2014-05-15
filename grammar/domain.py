import numbers
import numpy as np
import utils

class Domain(object):
    '''Base class'''
    def __init__(self, name, datatype):
        self.name = name
        self.datatype = datatype

class DiscreteDomain(Domain):
    def __init__(self, name, datatype):
        super(DiscreteDomain, self).__init__(name, datatype)

class NumericalDomain(Domain):
    epsilon = 0.000001
    def __init__(self, name, datatype, lower=None, upper=None):
        super(NumericalDomain, self).__init__(name, datatype)
        if not (self.valid(lower) and self.valid(upper)):
            raise ValueError('Bounds must be numbers or None: '
                             'lower %s, upper %s' % (lower, upper))
        if lower is None: self.lower = -np.inf
        else: self.lower = lower
        if upper is None: self.upper = np.inf
        else: self.upper = upper

    def valid(self, num):
        return isinstance(num, numbers.Number) or num is None

    def norm(self, numarray):
        return numarray

    def sample_mean_and_std(self, numarray, weights=None):
        if len(numarray) < 1:
            return None, None
        # utils.logger(numarray)
        # utils.logger(weights)
        w = np.where(np.logical_not(np.isnan(numarray)))
        numarray = numarray[w]
        if not weights is None: weights = weights[w]
        if sum(weights) == 0:
            return None, None
        # utils.logger(weights)
        mean = np.average(numarray, weights=weights)
        std = np.average((numarray-mean)**2, weights=weights)
        return mean, std+self.epsilon


class CircularDomain(NumericalDomain):
    '''For domains which wrap around'''

    def __init__(self, name, datatype, lower, upper):
        super(CircularDomain, self).__init__(name, datatype, lower=lower, upper=upper)
        if self.invalid(lower) or self.invalid(upper):
            raise ValueError('Invalid bounds, circular domains must be bounded:'
                             ' lower %s, upper %s' % (lower, upper))

    @staticmethod
    def invalid(num):
        return np.isnan(num) or num in (None, -np.inf, np.inf)

    def to_radians(self, numarray):
        return 2*np.pi*(numarray-self.lower)/(self.upper-self.lower)

    def from_radians(self, radarray):
        return self.norm((self.upper-self.lower)*radarray/(2*np.pi)+self.lower)

    def norm(self, numarray):
        return (numarray+self.upper)%(self.upper-self.lower)+self.lower

    @staticmethod
    def angular_mean_and_std(radarray, weights=None):
        if len(radarray) < 1 or sum(weights) == 0:
            return None, None
        # utils.logger(radarray)
        # utils.logger(weights)
        sin = np.average(np.sin(radarray), weights=weights)
        cos = np.average(np.cos(radarray), weights=weights)
        mean = np.arctan2(sin, cos)
        std = np.sqrt(-2*np.log(np.hypot(sin,cos)))
        return mean, std

    def sample_mean_and_std(self, numarray, weights=None):
        w = np.where(np.logical_not(np.isnan(numarray)))
        numarray = numarray[w]
        if not weights is None: weights = weights[w]
        mean, std = self.angular_mean_and_std(self.to_radians(numarray), 
                                              weights=weights)
        if mean is None:
            return mean, std
        else:
            return self.from_radians(mean), \
                   self.from_radians(std)-self.lower+self.epsilon