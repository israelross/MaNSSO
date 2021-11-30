import numpy as np
import numpy.linalg as la


class SmoothingManifoldManimizer:
    def __init__(self, manifold, f, g, transformation=None):
        self._A = transformation
        self._M = manifold
        self._f = f
        self._g = g
        if transformation is not None:
            self._alpha = f.L * manifold.M**2 + manifold.L * (g.D * la.norm(transformation, 2) + f.M)
            self._beta = (manifold.M * la.norm(transformation, 2)) ** 2
        else:
            self._alpha = f.L * manifold.M ** 2 + manifold.L * (g.D + f.M)
            self._beta = manifold.M ** 2

    def lipschitz_next_point(self, mu, x, v):
        # print((self._alpha + self._beta * (1/mu)))
        # return self._M.retract(x, (1 / (self._alpha + self._beta * (1/mu))) * v)
        return self._M.retract(x, (1 / 30) * v)

    def set_backtrack_params(self, factor, threshold, start):
        self._factor = factor
        self._threshold = threshold
        self._start = start

    def backtrack_next_point(self, mu, x, v):
        if self._A is not None:
            def perform(z):
                return self._f.perform(z) + self._g.moreau(self._A.dot(z), mu)
        else:
            def perform(z):
                return self._f.perform(z) + self._g.moreau(z, mu)
        original_value = perform(x)
        # print((perform(self._M.retract(x, v*0.01)) - perform(x)) / 0.01)
        point = self._M.retract(x, v*self._curr_bt)
        self.count = 1
        while perform(point) > original_value - self._threshold * self._curr_bt * (la.norm(v)**2):
            self._curr_bt = self._curr_bt * self._factor
            point = self._M.retract(x, v*self._curr_bt)
            self.count += 1
            if self.count == 50:
                print("backtrack procedure failed")
                return None
        return point

    def optimize(self, method='constant', max_iter=1000, mu_oracle='std', ret_iter_list=False, init_point=None, ret_counter=False):
        if method == 'constant':
            next_point = self.lipschitz_next_point
        elif method == 'backtrack':
            self._curr_bt = self._start
            next_point = self.backtrack_next_point
        else:
            raise Exception("Bad method")

        if mu_oracle == 'std':
            mu_oracle = lambda n: 0.01 * np.power(n, -1/3)

        if ret_iter_list:
            iter_list = []
        counter_list=[]

        if init_point is None:
            point = self._M.random_point()
        else:
            point = init_point
        self.count = 1
        for i in range(1, max_iter):
            mu = mu_oracle(i)
            if ret_iter_list:
                iter_list.append(point)
            if ret_counter:
                counter_list.append(self.count)
            if self._A is not None:
                v = -self._M.project_vector_to_tangent(point, self._f.grad(point) +
                                                              (1/mu) * (self._A.T.dot(self._A.dot(point) -
                                                                        self._g.prox(self._A.dot(point), mu))))
            else:
                v = -self._M.project_vector_to_tangent(point, self._f.grad(point) +
                                                       (1 / mu) * (point - self._g.prox(point, mu)))
                print(la.norm(v))
            new_point = next_point(mu, point, v)
            if new_point is None:
                break
            point = new_point

        if ret_iter_list:
            return np.array(iter_list), counter_list

        return point


class SubgradientManifoldMinimizer:
    def __init__(self, manifold, f, g, transformation=None):
        self._A = transformation
        self._M = manifold
        self._f = f
        self._g = g

    def optimize(self, max_iter=1000, ret_iter_list=False, init_point=None):
        if ret_iter_list:
            iter_list = []

        if init_point is None:
            point = self._M.random_point()
        else:
            point = init_point

        for i in range(1, max_iter):
            step = 1/np.sqrt(i)
            if ret_iter_list:
                iter_list.append(point)
            if self._A is not None:
                raise Exception("Error")
            else:
                v = -step * self._M.project_vector_to_tangent(point, self._f.grad(point) +
                                                                     self._g.subgrad(point))
            new_point = self._M.retract(point, v)
            if new_point is None:
                break
            point = new_point

        if ret_iter_list:
            return np.array(iter_list)

        return point

class MADMMinimizer:
    def __init__(self, manifold, f, g, rho, transformation=None):
        self._A = transformation
        self._M = manifold
        self._f = f
        self._g = g
        self._rho = rho

    def optimize(self, max_iter=1000, ret_iter_list=False, init_point=None):
        if ret_iter_list:
            iter_list = []

        if init_point is None:
            point = self._M.random_point()
        else:
            point = init_point

        x = point
        z = np.copy(x)
        u = np.zeros(x.shape)
        for i in range(1, max_iter):
            if ret_iter_list:
                iter_list.append(x)
            if self._A is not None:
                raise Exception("Error")
            else:
                v = -self._M.project_vector_to_tangent(x, self._f.grad(x) + self._rho * (x - z + u))
            x = self._M.retract(x, v)
            z = self._g.prox(x + u, 1/self._rho)
            u += x - z

        if ret_iter_list:
            return np.array(iter_list)
