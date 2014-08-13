
# Scipy imports.
from scipy import linalg, special
from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
     ravel, power, atleast_1d, squeeze, sum, transpose
import numpy 



class gaussian_kde(object):
    def __init__(self, dataset, weights=None):
        self.dataset = numpy.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        weights = numpy.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        inv_cov, norm_factor = Covariator(dataset, weights)()
        self.inv_cov = inv_cov
        self._norm_factor = norm_factor

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,), dtype=numpy.float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0) / 2.0
                result = result + self.weights[i]*exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = sum(self.weights*exp(-energy), axis=0)

        result = result / self._norm_factor

        return result

    __call__ = evaluate


class Covariator(object):
    def __init__(self, dataset, weights):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape
        weights = numpy.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        

    def scotts_factor(self):
        return power(numpy.ceil(self.n*(1-(self.weights**2).sum())), -1./(self.d+4))

    def silverman_factor(self):
        return power(numpy.ceil(self.n*(1-(self.weights**2).sum()))*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def __call__(self, bw_method=None):
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif numpy.isscalar(bw_method) and not isinstance(bw_method, basestring):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        return self._compute_covariance()


    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        self._data_covariance = atleast_2d(cov(self.dataset.T, self.weights))

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = linalg.pinv(self.covariance)
        self._norm_factor = (2*pi)**(self.d/2.)*sqrt(linalg.det(self.covariance))
        return self.inv_cov, self._norm_factor
        
def cov(data, weights):
  # gives biased cov. estimate
  # data.shape = n_elements, n_dimensions
  data, weights = numpy.array(data), numpy.array(weights, dtype=float)
  weights /= weights.sum()
  weights = weights[:,numpy.newaxis]
  mean = (data*weights).sum(axis=0)
  data -= mean
  return numpy.dot((data*weights).T, data)