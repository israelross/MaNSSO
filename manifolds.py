import numpy as np
import scipy as sp
import numpy.linalg as la

class Manifold:
    """
    This class represents an embedded submanifold $M$ of some Euclidean space.
    """
    def __init__(self):
        pass

    def project(self, point):
        """
        :param point: in the ambient Euclidean space
        :return: $\argmin_{x\in M} ||x-point||_2$
        """
        pass

    def project_vector_to_tangent(self, point, vector):
        """
        This function takes projects a vector on the tangent space of a given point
        :param point: on the manifold
        :param vector: in the ambient Euclidean space
        :return: $\argmin_{v \in T_{point} M} ||v - vector||_2$
        """
        pass

    def retract(self, point, vector):
        """
        This function must satisfy that retract(point, 0) == point and
        d retract(point, vector)
        ------------------------ = ID_{T_{point}} M
                d vector
        :param point: on the manifolfdd
        :param vector: in $T_{point} M$
        :return:
        """
        pass

    def random_point(self):
        """
        :return: random point on $M$.
        """
        pass


class ManifoldWithProjectionRetraction(Manifold):
    """
    Calculates automatically the retraction
    """
    def __init__(self):
        super(ManifoldWithProjectionRetraction, self).__init__()
        self.M = None
        self.L = None
        pass

    def project(self, point):
        pass

    def project_vector_to_tangent(self, point, vector):
        pass

    def retract(self, point, vector):
        return self.project(point + self.project_vector_to_tangent(point, vector))


class StiefelManifold(ManifoldWithProjectionRetraction):
    def __init__(self, d, p):
        super(StiefelManifold, self).__init__()
        self._d = d
        self._p = p
        self.M = 1
        self.L = np.sqrt(p + 1) + 0.5

    def project(self, point):
        if point.shape != (self._d, self._p):
            raise "Wrong shape of point"
        if self._p == 1:
            return point/la.norm(point)
        u, _, v = la.svd(point, full_matrices=True)
        return u.dot(np.eye(self._d, self._p)).dot(v)

    def project_vector_to_tangent(self, point, vector):
        if point.shape != (self._d, self._p) or vector.shape != (self._d, self._p):
            raise "Wrong shape of point"
        skew = 0.5 * (point.T.dot(vector) - vector.T.dot(point))
        return (np.eye(self._d) - point.dot(point.T)).dot(vector) + point.dot(skew)

    def random_point(self):
        return self.project(np.random.normal(0, 1, (self._d, self._p)))