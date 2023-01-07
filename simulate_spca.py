import numpy as np

from manifolds import *
from functions import *
from minimization import *
import pickle


def spca_cov_given_var_matrix(d, p, var):
    V = np.zeros((d, p))
    if p <= 4:
        n_blocks = 4
    else:
        n_blocks = 8
    for block in range(n_blocks):
        V[block * d // 8: (block + 1) * d // 8, block * p // n_blocks:(block + 1) * p // n_blocks] = \
            StiefelManifold(d // 8, p // n_blocks).random_point()
    A = V.dot(var * np.eye(p)).dot(V.T) + np.eye(d)
    N = np.diag(1 / np.sqrt(np.diag(A)))
    return V, N.dot(A).dot(N)


class run_results:
    def __init__(self, times, vals, name, last_point=None):
        self._times = np.array(times)
        self._vals = np.array(vals)
        self._last_point = last_point
        self._best_val = np.min(self._vals)
        self._name = name

    def truncate_at_time(self, T):
        return run_results(self._times[self._times < T], self._vals[self._times < T], self._name)

    def time_to_val(self, val):
        inds = np.where(self._vals < val)[0]
        if len(inds) == 0:
            return np.inf
        return self._times[inds[0]]

    def plot(self, fig, label=-1):
        if label == -1:
            label = self._name
        fig.plot(self._times, self._vals, label=label)


class simulator_solver:
    def __init__(self, p, var, time_limit, d=1024, lam=1):
        self._p = p
        self._d = d
        self._var = var
        self._lam = lam
        self._time_limit = time_limit
        self._V, self._cov = spca_cov_given_var_matrix(d, p, var)
        self._M = StiefelManifold(self._d, self._p)
        self._f = L2PCA(self._cov, self._p, 'cov')
        self._g = L1Norm(self._cov.shape[0], self._p, self._lam)
        self._results = {}
        self._init_point = self._M.random_point()
        self._init_val = self._f.perform(self._init_point) + self._g.perform(self._init_point)
        self._bests = {}

    def run_method(self, alg, oracle, name):
        name = alg + '_' + name
        if alg == 'A2':
            sminimizer = DSGMMnimizer(self._M, self._f, self._g)
            sminimizer.set_backtrack_params(factor=0.5, threshold=0.5, start=1)
            res, times = sminimizer.optimize(max_iter=np.inf,
                                                              mu_oracle=oracle,
                                                              stepsize_strategy='backtrack', ret_iter_list=True,
                                                              time_limit=self._time_limit, init_point=self._init_point)

        if alg == 'RSG':
            gminimizer = SubgradientManifoldMinimizer(self._M, self._f, self._g)
            res, times = gminimizer.optimize(max_iter=np.inf, ret_iter_list=True, time_limit=self._time_limit,
                                                 gamma_oracle=oracle, init_point=self._init_point)
        self._results[name] = run_results(times,
                                          [self._f.perform(i) + self._g.perform(i) for i in res],
                                          name,
                                          res[-1])
        self._bests[name] = self._results[name]._best_val

    def best_method(self):
        return min(self._bests, key=self._bests.get)

    def best_val(self):
        return min(self._bests.values())

    def quality_val(self, p):
        return self._init_val - p * (self._init_val - self.best_val())

    def get_time_to_quality_val(self, p):
        val = self.quality_val(p)
        return {name: self._results[name].time_to_val(val) for name in self._results}\

    def truncate(self, T):
        ss = simulator_solver(self._p, self._var, T)
        ss._results = {method: self._results[method].truncate_at_time(T) for method in self._results}
        ss._bests = {method: ss._results[method]._best_val for method in self._results}
        return ss

    def plot(self, fig):
        for method in self._results:
            self._results[method].plot(fig)



if __name__ == "__main__":
    TIME_LIMIT = 2
    REPS = 20
    A2_ORACLES = {
        '0.1_exp0.33': lambda i: 0.1 / np.power(i, 1/3),
        '0.1_exp0.66': lambda i: 0.1 / np.power(i, 2/3),
        '0.1_exp0.5': lambda i: 0.1 / np.power(i, 1/2)
    }
    RSG_EXP_ORACLES = {
        '0.7exp': lambda i: np.power(0.7, i),
        '0.8exp': lambda i: np.power(0.8, i),
        '0.9exp': lambda i: np.power(0.9, i)
    }
    RSG_SQRT_ORACLES = {
        'sqrt_1': lambda i: 1 / np.sqrt(i),
        'sqrt_0.1': lambda i: 0.1 / np.sqrt(i),
        'sqrt_0.01': lambda i: 0.01 / np.sqrt(i)
    }
    for p in [8, 16, 24, 32, 40]:
        for var in [2, 4, 8, 16, 32]:
            l = []
            for i in range(REPS):
                ss = simulator_solver(p, var, TIME_LIMIT)
                for o in A2_ORACLES:
                    ss.run_method('A2', A2_ORACLES[o], o)
                for o in RSG_EXP_ORACLES:
                    ss.run_method('RSG', RSG_EXP_ORACLES[o], o)
                for o in RSG_SQRT_ORACLES:
                    ss.run_method('RSG', RSG_SQRT_ORACLES[o], o)
                l.append(ss)
            with open('SPCA_last/run_%i_%i.pkl' % (p, var), 'wb') as f:
                pickle.dump(l, f)
            print('finished %i : %i' % (p, var))
