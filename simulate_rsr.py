from manifolds import *
from functions import *
from minimization import *
import pickle
from scipy.linalg import subspace_angles, null_space


def sphere_rand(d):
    x = np.random.normal(0, 1, d)
    return x / la.norm(x)


def generate_y_matrix(m, n, r, m_inlier):
    subspace_mat = StiefelManifold(n, n-r).random_point()
    m_outlier = m - m_inlier
    y = np.array([subspace_mat.dot(sphere_rand(n-r)) for _ in range(m_inlier)] +
                 [sphere_rand(n) for _ in range(m_outlier)])
    return np.array(y), subspace_mat


n = 100
r_pool = [5, 10, 15, 20]
m = 5000
m_inliers_pool = [250, 500, 1000, 1500, 2000]
results = {}
REPS = 50
for m_inliers in m_inliers_pool:
    for r in r_pool:
        results[r, m_inliers] = []
        for i in range(REPS):
            results[r, m_inliers].append({})
            curr_dic = results[r, m_inliers][-1]
            y, subspace_mat = generate_y_matrix(m, n, r, m_inliers)
            true_r_sub = null_space(subspace_mat.T)

            curr_dic['y'] = y
            curr_dic['true_r_sub'] = true_r_sub

            a2_optimizer = DSGMMnimizer(StiefelManifold(n, r), f_zero(), L21Norm(m, r), y)
            a2_optimizer.set_backtrack_params(factor=0.5, threshold=0.5, start=1)
            a2_res, a2_times = a2_optimizer.optimize(max_iter=np.inf, mu_oracle=lambda t: 0.1 / np.power(t, 2 / 3),
                                                         stepsize_strategy='backtrack', ret_iter_list=True,
                                                         time_limit=1)
            a2_pers = np.array([L21Norm(m, r).perform(y.dot(x)) for x in a2_res])
            a2_dists = np.array([r - sum(la.svd(x.T.dot(true_r_sub))[1]) for x in a2_res])

            rsg_optimizer = SubgradientManifoldMinimizer(StiefelManifold(n, r), f_zero(), L21Norm(m, r), y)
            rsg_res, rsg_times = rsg_optimizer.optimize(max_iter=np.inf, ret_iter_list=True, time_limit=1,
                                                         gamma_oracle=lambda t: 0.1 * (0.75**t), init_point=a2_res[0])
            rsg_pers = np.array([L21Norm(m, r).perform(y.dot(x)) for x in rsg_res])
            rsg_dists = np.array([r - sum(la.svd(x.T.dot(true_r_sub))[1]) for x in rsg_res])

            curr_dic['a2_2/3_optimizer'] = a2_optimizer
            curr_dic['a2_2/3_last_iter'] = a2_res[-1]
            curr_dic['a2_2/3_pers'] = a2_pers
            curr_dic['a2_2/3_times'] = a2_times
            curr_dic['a2_2/3_dists'] = a2_dists

            curr_dic['rsg_0.75_optimizer'] = rsg_optimizer
            curr_dic['rsg_0.75_last_iter'] = rsg_res[-1]
            curr_dic['rsg_0.75_pers'] = rsg_pers
            curr_dic['rsg_0.75_times'] = rsg_times
            curr_dic['rsg_0.75_dists'] = rsg_dists

            a2_optimizer = DSGMMnimizer(StiefelManifold(n, r), f_zero(), L21Norm(m, r), y)
            a2_optimizer.set_backtrack_params(factor=0.5, threshold=0.5, start=1)
            a2_res, a2_times = a2_optimizer.optimize(max_iter=np.inf, mu_oracle=lambda t: 0.1 / np.power(t, 1 / 2),
                                                     stepsize_strategy='backtrack', ret_iter_list=True,
                                                     time_limit=1)
            a2_pers = np.array([L21Norm(m, r).perform(y.dot(x)) for x in a2_res])
            a2_dists = np.array([r - sum(la.svd(x.T.dot(true_r_sub))[1]) for x in a2_res])

            rsg_optimizer = SubgradientManifoldMinimizer(StiefelManifold(n, r), f_zero(), L21Norm(m, r), y)
            rsg_res, rsg_times = rsg_optimizer.optimize(max_iter=np.inf, ret_iter_list=True, time_limit=1,
                                                        gamma_oracle=lambda t: 0.1 * (0.9 ** t), init_point=a2_res[0])
            rsg_pers = np.array([L21Norm(m, r).perform(y.dot(x)) for x in rsg_res])
            rsg_dists = np.array([r - sum(la.svd(x.T.dot(true_r_sub))[1]) for x in rsg_res])

            curr_dic['a2_1/2_optimizer'] = a2_optimizer
            curr_dic['a2_1/2_last_iter'] = a2_res[-1]
            curr_dic['a2_1/2_pers'] = a2_pers
            curr_dic['a2_1/2_times'] = a2_times
            curr_dic['a2_1/2_dists'] = a2_dists

            curr_dic['rsg_0.9_optimizer'] = rsg_optimizer
            curr_dic['rsg_0.9_last_iter'] = rsg_res[-1]
            curr_dic['rsg_0.9_pers'] = rsg_pers
            curr_dic['rsg_0.9_times'] = rsg_times
            curr_dic['rsg_0.9_dists'] = rsg_dists

with open('rsr_results2.pkl', "wb") as f:
    pickle.dump(results, f)
