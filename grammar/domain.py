import numbers
import numpy as np

class Domain(object):
    '''Base class'''
    def __init__(self, name, lower=None, upper=None):
        if not (self.valid(lower) and self.valid(upper)):
            raise ValueError('Bounds must be numbers or None: '
                             'lower %s, upper %s' % (lower, upper))
        if lower is None: self.lower = -np.inf
        else: self.lower = lower
        if upper is None: self.upper = np.inf
        else: self.upper = upper

    def valid(self, num):
        return isinstance(num, numbers.Number) or num is None

    def sample_mean_and_std(self, numarray):
        mean = numarray.mean()
        std = np.sqrt(((numarray-mean)**2).sum()/(numarray.shape[0]-1))
        return mean, std


class CircularDomain(Domain):
    '''For domains which wrap around'''

    def __init__(self, name, lower, upper):
        super(CircularDomain, self).__init__(name, lower=lower, upper=upper)
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
    def angular_mean_and_std(radarray):
        sin = np.mean(np.sin(radarray))
        cos = np.mean(np.cos(radarray))
        mean = np.arctan2(sin, cos)
        std = np.sqrt(-2*np.log(np.hypot(sin,cos)))
        return mean, std

    def sample_mean_and_std(self, numarray):
        mean, std = self.angular_mean_and_std(self.to_radians(numarray))
        return self.from_radians(mean), self.from_radians(std)-self.lower