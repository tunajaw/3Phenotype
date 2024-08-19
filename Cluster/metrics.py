import networkx as nx
import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    auc,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph

from Util.utils import EPS

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets, dtype=int).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

# subfunctions of calculate_MI

def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    # dist = dist/torch.max(dist)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    L, Q = torch.torch.linalg.eigh(k)  # pylint: disable=unused-variable  # type: ignore
    eigv = torch.abs(L)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    L, Q = torch.torch.linalg.eigh(k)  # pylint: disable=unused-variable  # type: ignore
    eigv = torch.abs(L)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    # normlize = Ixy/(torch.max(Hx,Hy))

    return Ixy

# cluster metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)  # pyright: ignore


def f_get_prediction_scores(y_true, y_pred):
    if len(np.unique(y_true)) != 2:  # no label for running roc_auc_curves
        auroc = -1.0
        auprc = -1.0
    else:
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
    return (auroc, auprc)


def get_auc_scores(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = y_true[mask == 1]
        y_pred = y_pred[mask == 1]
    y_dim = y_true.shape[-1]
    AUROC = np.zeros([y_dim])
    AUPRC = np.zeros([y_dim])
    for i in range(y_dim):
        y_true_i = y_true[..., i].reshape((-1,))
        y_pred_i = y_pred[..., i].reshape((-1,))
        AUROC[i], AUPRC[i] = f_get_prediction_scores(y_true_i, y_pred_i)

    return AUROC, AUPRC


def get_cls_scores(*args, **kwargs):  # pylint: disable=unused-argument
    c_pred = kwargs.get("c_pred", None)
    c_true = kwargs.get("c_true", None)
    if c_true is not None:
        score_with_label = get_cls_scores_from_label(c_true, c_pred)
    else:
        score_with_label = {}

    x = kwargs.get("x", None)
    y_true = kwargs.get("y_true", None)
    if x is not None and c_pred is not None and y_true is not None:
        score_without_label = get_cls_scores_without_label(x, c_pred, y_true)
    else:
        score_without_label = {}

    scores = {**score_with_label, **score_without_label}
    return scores


# 1. with ground truth
def get_cls_scores_from_label(c_true, c_pred):
    scores = {}

    purity = purity_score(c_true, c_pred)
    rand = adjusted_rand_score(c_true, c_pred)
    mutual_info = normalized_mutual_info_score(c_true, c_pred)
    scores["PURITY"] = purity
    scores["RAND"] = rand
    scores["MI"] = mutual_info
    return scores


# 2. without ground truth
def get_cls_scores_without_label(x, c_pred, y_true):  # pylint: disable=unused-argument
    scores = {}
    # eval = evaluate_MI(x, c_pred, y_true)
    # scores['I(C,y)'] = eval['I(C,Y)']
    # scores['H(C)'] = eval['mean[H(C)]']
    batch_size, _, x_dim = x.shape  # pylint: disable=unused-variable
    x = x.reshape((batch_size, -1))
    print('evaluate knn score:')
    scores["Silhouette_knn"] = evaluate_silhouette(x, c_pred, y=None, topk=10)
    print('evaluate auc score:')
    scores["Silhouette_auc"] = get_silhouette_auc(x, c_pred)
    return scores


def evaluate_MI(x, c, y):
    # x: test_size x x_dim
    # c: test_size
    # y: test_size x y_dim
    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)
    n = int(np.max(c)) + 1
    c_one_hot = get_one_hot(c, n)
    C = torch.from_numpy(c_one_hot)
    I_CY = calculate_MI(C, Y, 0.1, 0.1).item()
    H_X = calculate_MI(X, X, 0.1, 0.1).item()  # noqa F841 # pylint: disable=unused-variable
    cs = np.unique(c)
    H_C = np.zeros((len(cs),))
    for i, c_i in enumerate(cs):
        X_C = X[c == c_i]
        H_C[i] = calculate_MI(X_C, X_C, 0.1, 0.1).item()
    info = {}
    info["mean[H(C)]"] = np.mean(H_C)
    info["H(C)"] = H_C
    info["I(C,Y)"] = I_CY
    return info


def evaluate_silhouette(x, c, y=None, topk=None):
    # x: test_size x x_dim
    # c: test_size
    # y: test_size x y_dim
    if y is not None:
        x = np.concatenate([x, y], axis=-1)

    try:
        if topk is not None:
            score = silhouette_score_knn(x, c, topk=topk)
        else:
            score = silhouette_score(x, c)
    except Exception:  # pylint: disable=broad-exception-caught
        score = np.nan
    return score


def get_silhouette_auc(x, c):
    try:
        conn, silh = get_silhouette_curve(x, c)
        score = auc(conn, silh)
    except Exception:  # pylint: disable=broad-exception-caught
        score = np.nan
    return score


def get_silhouette_curve(x, c, k_max=20):
    conn = np.array([connectivity_score_knn(x, c, topk=k) for k in np.arange(1, k_max)])
    silh = np.array([silhouette_score_knn(x, c, topk=k) for k in np.arange(1, k_max)])
    return conn, 0.5 * (silh + 1)


def connectivity_score_knn(x, c, topk=None):
    c_vals, counts = np.unique(c, return_counts=True)
    scores = np.zeros((len(c_vals),))
    for i, c_val in enumerate(c_vals):
        if counts[i] == 1:
            scores[i] = 1.0
        else:
            X = x[c == c_val]
            if topk is None:
                topk = len(X) - 1
            k = min(topk, len(X) - 1)
            A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
            n_components = nx.number_connected_components(nx.from_numpy_array(A))
            scores[i] = 1.0 / n_components

    return np.mean(scores)


def silhouette_score_knn(x, c, topk=None):
    c_vals, counts = np.unique(c, return_counts=True)
    if len(c_vals) < 2:
        raise ValueError(f"number of labels is {len(c_vals)} < 2")

    dist = euclidean_distances(x)
    indicies = np.arange(len(c))

    dist_table = []
    for val in c_vals:
        mask = c == val
        c_dist = dist[:, mask]
        n, c_size = c_dist.shape  # pylint: disable=unused-variable
        idx_c = indicies[mask]

        if topk is not None:
            kth = min(c_size - 1, topk)
            top_k_idx = np.argpartition(c_dist, kth=kth, axis=-1)[:, :kth]
            idx_knn = np.take(idx_c, top_k_idx, axis=0)
            x_knn = np.take(x, idx_knn, axis=0)
            diff = x[:, np.newaxis, :] - x_knn
            diff = np.sum(diff, axis=1) / (kth - 1.0 * mask[:, np.newaxis] + EPS)
            min_dist = np.linalg.norm(diff, ord=2, axis=-1)
            # c_dist_top_k = np.partition(c_dist, kth=kth, axis=-1)[:, :kth]
            # min_dist = np.sum(c_dist_top_k, axis=-1) / (kth - 1.0 * mask + EPS)    # ignore d(xi,xi) -- mask == 1
        else:
            min_dist = np.sum(c_dist, axis=-1) / (c_size - 1.0 * mask + EPS)  # ignore d(xi,xi) -- mask == 1

        dist_table.append(min_dist)
    dist_table = np.stack(dist_table, axis=0)

    a = np.zeros_like(c)
    b = np.zeros_like(c)

    for i, val in enumerate(c_vals):
        mask = c == val
        a[mask] = dist_table[i][mask]
        b[mask] = np.min(dist_table[c_vals != val], axis=0)[mask]

    ab = np.stack([a, b], axis=-1)

    s = (ab[:, 1] - ab[:, 0]) / (np.max(ab, axis=-1) + EPS)

    for i, val in enumerate(c_vals):
        if counts[i] == 1:
            s[c == val] = 0

    return np.mean(s)
