# references: [https://github.com/yvchao/tphenotype]


import numpy as np

EPS = 1e-6

def plogp_q(p, q):
    return p * (np.log(p + EPS) - np.log(q + EPS))


def d_kl(p, q):
    # p,q: ... x y_dim for categorical distribution
    # d: ...
    d = np.sum(plogp_q(p, q), axis=-1)
    return d

def batch_d(p, q):
    # ... x batch_size_p x batch_size_q x y_dim
    m = 0.5 * (p[..., :, np.newaxis, :] + q[..., np.newaxis, :, :])
    # ... x batch_size_p x batch_size_q
    d_p_m = d_kl(p[..., :, np.newaxis, :], m)
    d_q_m = d_kl(q[..., np.newaxis, :, :], m)
    d = 0.5 * (d_p_m + d_q_m)
    return d