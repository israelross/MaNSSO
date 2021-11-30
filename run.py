from manifolds import *
from functions import *
from minimization import *
import matplotlib.pyplot as plt
import pickle

def zero_small_vals(mat, threshold):
    x = np.zeros(mat.shape)
    x[np.abs(mat)>threshold] = mat[np.abs(mat)>threshold]
    return x

cov = np.array([[1.000,  0.954,  0.364,  0.342, -0.129,  0.313,  0.496,  0.424,  0.592,  0.545,  0.084, -0.019,  0.134],
                [0.954,  1.000,  0.297,  0.284, -0.118,  0.291,  0.503,  0.419,  0.648,  0.569,  0.076, -0.036,  0.144],
                [0.364,  0.297,  1.000,  0.882, -0.148,  0.153, -0.029, -0.054,  0.125, -0.081,  0.162,  0.220,  0.126],
                [0.342,  0.284,  0.882,  1.000,  0.220,  0.381,  0.174, -0.059,  0.137, -0.014,  0.097,  0.169,  0.015],
                [0.129, -0.118, -0.148,  0.220,  1.000,  0.364,  0.296,  0.004, -0.039,  0.037, -0.091, -0.145, -0.208],
                [0.313,  0.291,  0.153,  0.381,  0.364,  1.000,  0.813,  0.090,  0.211,  0.274, -0.036,  0.024, -0.329],
                [0.496,  0.503, -0.029,  0.174,  0.296,  0.813,  1.000,  0.372,  0.465,  0.679, -0.113, -0.232, -0.424],
                [0.424,  0.419, -0.054, -0.059,  0.004,  0.090,  0.372,  1.000,  0.482,  0.557,  0.061, -0.357, -0.202],
                [0.592,  0.648,  0.125,  0.137, -0.039,  0.211,  0.465,  0.482,  1.000,  0.526,  0.085, -0.127, -0.076],
                [0.545,  0.569, -0.081, -0.014,  0.037,  0.274,  0.679,  0.557,  0.526,  1.000, -0.319, -0.368, -0.291],
                [0.084,  0.076,  0.162,  0.097, -0.091, -0.036, -0.113,  0.061,  0.085, -0.319,  1.000,  0.029,  0.007],
                [0.019, -0.036,  0.220,  0.169, -0.145,  0.024, -0.232, -0.357, -0.127, -0.368,  0.029,  1.000,  0.184],
                [0.134,  0.144,  0.126,  0.015, -0.208, -0.329, -0.424, -0.202, -0.076, -0.291,  0.007,  0.184,  1.000]]
               )
np.set_printoptions(precision=2)
zou_res = np.array([[-0.42,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [-0.42,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.64,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.54,  0.42,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.58,  0.  ,  0.  ,  0.  ],
       [-0.3 ,  0.  ,  0.57,  0.  ,  0.  ,  0.  ],
       [-0.42,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [-0.3 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [-0.37,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [-0.39,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ],
       [ 0.  ,  0.41,  0.  ,  0.  ,  1.  ,  0.  ],
       [ 0.  ,  0.36, -0.39,  0.  ,  0.  ,  1.  ]])


class SPCAComparingMinimizers:
    def __init__(self, cov, p, lam):
        print(p)
        self._d = cov.shape[0]
        self._M = StiefelManifold(self._d, p)
        self._f = L2PCA(cov, p, 'cov')
        self._g = L1Norm(cov, p, lam)
        self._sminimizer = SmoothingManifoldManimizer(self._M, self._f, self._g)
        self._gminimizer = SubgradientManifoldMinimizer(self._M, self._f, self._g)
        self._sminimizer.set_backtrack_params(factor=0.5, threshold=0.5, start=1)

    def minimize(self, iters=1000, plot=True):
        self.s1_res, self.s1_cnt = self.sminimize(ret_iter_list=True, max_iter=iters)
        self.s2_res, self.s2_cnt = self.sminimize(ret_iter_list=True, max_iter=iters, mu_oracle=lambda i: 0.1*np.power(i, -1 / 2))
        self.g_res = self.gminimize(ret_iter_list=True, max_iter=iters)
        self.s1_fun = np.array([minimizer.perform(i) for i in self.s1_res])
        self.s2_fun = np.array([minimizer.perform(i) for i in self.s2_res])
        self.g_fun = np.array([minimizer.perform(i) for i in self.g_res])
        if plot:
            plt.plot(np.cumsum(self.s1_cnt), self.s1_fun, label=r'Algorithm 2, $\mu_k=\frac{1}{k^{1/3}}$')
            plt.plot(np.cumsum(self.s2_cnt), self.s2_fun, label=r'Algorithm 2, $\mu_k=\frac{0.1}{\sqrt{k}}$')
            plt.plot(range(len(self.g_fun)), self.g_fun, label='RSG')

    def sminimize(self, mu_oracle=lambda i: 1*np.power(i, -1 / 3), max_iter=1000, ret_iter_list=False):
        return self._sminimizer.optimize(max_iter=max_iter, mu_oracle=mu_oracle,
                                        stepsize_strategy='backtrack', ret_iter_list=ret_iter_list, ret_counter=True)

    def gminimize(self, max_iter=1000, ret_iter_list=False):
        return self._gminimizer.optimize(max_iter=max_iter, ret_iter_list=ret_iter_list)

    def perform(self, x):
        return self._f.perform(x) + self._g.perform(x)

    def expvar(self, x):
        return self._f.perform(x)

f = open('isolet1X.pickle', 'rb')
data = pickle.load(f)
cov = np.corrcoef(data.T)
minimizer = SPCAComparingMinimizers(cov, 15, 0.5)
minimizer.minimize(iters=1000, plot=True)

plt.xlabel("#retractions")
plt.ylabel("cost")
plt.legend()
plt.show()