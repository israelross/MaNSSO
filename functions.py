import numpy as np
import numpy.linalg as la


class SmoothFunction:
    """
    An abstract class of smooth function f on an Euclidean space
    """
    def __init__(self):
        self.L = None
        self.M = None
        pass

    def perform(self, x):
        """
        :param x: a point in the dimension of the input of f
        :return: f(x)
        """
        pass

    def grad(self, x):
        """
        :param x: a point in the dimension of the input of g
        :return: $\nabla f(x)$
        """
        pass


class L2PCA(SmoothFunction):
    """
    This function is
    $$
    f(A) = -Tr(A A^T X X^T)
    $$
    where $X$ is the data matrix. In fact in this function is initialized either with $X$ or with $X X^T$.
    """
    def __init__(self, data_or_cov, p, load_type='data'):
        super(L2PCA, self).__init__()
        if load_type == 'data':
            self._cov = data_or_cov.T.dot(data_or_cov)
        elif load_type == 'cov':
            self._cov = data_or_cov
        else:
            raise "Wrong load_type"
        self._p = p
        self._cov_norm = la.norm(self._cov, 2)
        self.L = 2 * self._cov_norm
        self.M = 2 * np.sqrt(p) * self._cov_norm

    def perform(self, x):
        return -np.sum(x * self._cov.dot(x))
        # return -np.trace(self._cov.dot(x.dot(x.T)))

    def grad(self, x):
        return -2 * self._cov.dot(x)


class LDA(SmoothFunction):
    def __init__(self, data, y):
        super(LDA, self).__init__()
        self.classed_data = [data[y==z] for z in np.unique(y)]
        self.class_means = [np.mean(data_of_class, axis=0) for data_of_class in self.classed_data]
        self.mean = np.mean(data, axis=0)
        self.Sw = np.sum([np.sum([(d[np.newaxis] - self.class_means[i]).T.dot(d[np.newaxis]-self.class_means[i])
                        for d in self.classed_data[i]], axis=0) for i in range(len(self.classed_data))], axis=0)
        self.Sb = np.sum([
            len(self.classed_data[i]) *
            (self.means[i][np.newaxis] - self.mean[np.newaxis]).T.dot(self.class_means[i][np.newaxis] - self.mean[np.newaxis])
            for i in range(len(self.means))
        ], axis=0)

    def perform(self, x):
        return np.trace(x.T.dot(self.Sb).dot(x)) / np.trace(x.T.dot(self.Sw).dot(x))

    def grad(self, x):
        f, g = np.trace(x.T.dot(self.Sb).dot(x)),  np.trace(x.T.dot(self.Sw).dot(x))
        return (x.T.dot(self.Sb + self.Sb.T) - (f/g) * x.T.dot(self.Sw + self.Sw.T))/g


class f_zero(SmoothFunction):
    """
    Constant zero function
    """
    def __init__(self):
        super(f_zero, self).__init__()
        self.L = 0
        self.M = 0

    def perform(self, x):
        return 0

    def grad(self, x):
        return np.zeros(x.shape)


class NonSmoothConvexFunction:
    """
    An abstract class for a non-smooth convex prox-tractable function
    """
    def __init__(self):
        self.D = None
        pass

    def perform(self, x):
        """
        :param x: a point in the dimension of the input of g
        :return: g(x)
        """
        pass

    def prox(self, x, mu):
        """
        :param x: a point in the dimension of the input of g
        :param mu: > 0
        :return: prox_{\mu g} (x)
        """
        pass

    def moreau(self, x, mu):
        """
        :param x:  a point in the dimension of the input of g
        :param mu: > 0
        :return: M_g ^\mu (x)
        """
        y = self.prox(x, mu)
        return (la.norm(x-y)**2)/(2*mu) + self.perform(y)


class L1Norm(NonSmoothConvexFunction):
    """
    This fdnction is $\lambda$ times the $\ell_1$ norm of the input.
    """
    def __init__(self, d, p, lam):
        super(L1Norm, self).__init__()
        self.D = d * p * lam
        self._lambda = lam

    def perform(self, x):
        return self._lambda * np.sum(np.abs(x))

    def prox(self, x, mu):
        res = np.zeros(x.shape)
        factor = mu * self._lambda
        res[x > factor] = x[x > factor] - factor
        res[x < -factor] = x[x < -factor] + factor
        return res

    def subgrad(self, x):
        return self._lambda * np.sign(x)


class L21Norm(NonSmoothConvexFunction):
    def __init__(self, m, r):
        super(L21Norm, self).__init__()
        self.m = m
        self.r = r

    def perform(self, x):
        return np.sum(la.norm(x, axis=1))

    def prox(self, x, mu):
        norms = la.norm(x, axis=1)
        return ((1 - mu / (np.maximum(mu, norms))) * x.T).T

    def subgrad(self, x):
        norms = la.norm(x, axis=1)
        return (x.T / norms).T


class L1NormSubsetIndices(NonSmoothConvexFunction):
    """
    This fdnction is $\lambda$ times the $\ell_1$ norm of the input.
    """
    def __init__(self, d, p, inds):
        super(L1NormSubsetIndices, self).__init__()
        self.D = d * p * len(inds[0])
        self._inds = inds

    def perform(self, x):
        return np.sum(np.abs(x[self._inds]))

    def prox(self, x, mu):
        res = np.array(x)
        y = x[self._inds]
        y[y > mu] -= mu
        y[y < -mu] += mu
        res[self._inds] = y
        return res

    def subgrad(self, x):
        res = np.zeros(x.shape)
        res[self._inds] = np.sign(res[self._inds])
        return res
