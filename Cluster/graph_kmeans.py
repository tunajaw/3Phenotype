# references: [https://github.com/yvchao/tphenotype]

import os
import pickle
import numpy as np
from tqdm import auto

from Util.utils import batch_d
from Util.decorators import timeit
from Util.big_data_array import BigArray, update_by_threshold
from concurrent.futures import ThreadPoolExecutor

def d_js(p, q):
    if len(p.shape) == 1:
        p = p[np.newaxis, :]

    if len(q.shape) == 1:
        q = q[np.newaxis, :]

    d = batch_d(p, q)
    return d


class GraphKmeans:
    def __init__(self, K, S, Q, chunked_rows, big_data=True, S_link='./temp_model/S', G_link = './temp_model/G'):
        N, y_dim = Q.shape
        self.K = K
        
        self.Q = Q
        self.delta = np.log(2)
        if big_data:
            self.S = None
            self.BigS = dict() # BigArray(chunked_rows, S_link)
            self.BigG = dict() # BigArray(chunked_rows, G_link)
        else:
            self.S = S
            self.BigS = None
            self.BigG = None
        
        self.N = N
        self.big_data = big_data

        self.cluster_assignment = np.full((N,), -1)
        self.cluster_centroids = np.full((self.K, y_dim), np.nan)

        training_dataset_cfgs = {'dataset':'training', 'chunked_rows':chunked_rows}
        self._init_BigArray(training_dataset_cfgs)

    def _init_BigArray(self, dataset_cfgs):
        link = f"./temp_model/kmeans/{dataset_cfgs['dataset']}"
        self.BigS[dataset_cfgs['dataset']] = BigArray(dataset_cfgs['chunked_rows'], f"{link}/A")
        self.BigG[dataset_cfgs['dataset']] = BigArray(dataset_cfgs['chunked_rows'], f"{link}/G")

    def _calculate_delta_J(self, i, K, cluster_assignment, S, BigS, big_data):
        delta_J = np.zeros(K)
        if cluster_assignment[i] != -1:
            return i, -1  # Skip updating this index
        for k in range(K):
            idx = np.where(cluster_assignment == k)[0]
            if big_data:
                delta_J[k] = 2 * BigS.big_sum([i], idx, along_axis=False)
            else:
                delta_J[k] = 2 * np.sum(S[i, idx])
        return i, np.argmin(delta_J)


    def initialize_centers(self, S, K):
        # # S: N x N distance matrix, symmetrical
        # N, _ = S.shape
        # print('S0')
        # cluster_assignment = np.full((self.N,), -1)

        # core_indicies = []
        # # cluster 1
        # if(self.big_data):
        #     total_distance = self.BigS['training'].big_sum(range(self.N), range(self.N))
        # else:
        #     total_distance = np.sum(S, axis=-1)
        # # the index which can be centroid
        # idx = np.argmin(total_distance)
        # cluster_assignment[idx] = 0
        # core_indicies.append(idx)
        # sample_idx = np.arange(self.N)
        # print('S1')
        # for k in range(1, K):
        #     mask = np.isin(sample_idx, core_indicies)
        #     candidate_idx = sample_idx[~mask]
        #     if(self.big_data):
        #         total_distance = self.BigS['training'].big_sum(candidate_idx, core_indicies)
        #     else:
        #         total_distance = np.sum(S[candidate_idx][:, core_indicies], axis=-1)

            
        #     idx = np.argmax(total_distance)
        #     sample_sel = candidate_idx[idx]
        #     cluster_assignment[sample_sel] = k
        #     core_indicies.append(sample_sel)
        # print('S2')
        # [TODO] Add an option to cancel warm start
        # with ThreadPoolExecutor(max_workers=24) as executor:
        #     futures = [executor.submit(self._calculate_delta_J, i, K, cluster_assignment, S, self.BigS['training'], self.big_data) 
        #             for i in range(len(cluster_assignment))]
        #     for future in auto.tqdm(futures):
        #         i, cluster = future.result()
        #         if cluster != -1:
        #             cluster_assignment[i] = cluster
        # for i, cluster in auto.tqdm(enumerate(cluster_assignment)):
        #     if cluster != -1:
        #         continue
        #     delta_J = np.zeros((K,))
        #     for k in range(K):
        #         (idx,) = np.where(cluster_assignment == k)
        #         if(self.big_data):
        #             delta_J[k] = 2 * self.BigS['training'].big_sum([i], idx, along_axis=False)
        #         else:
        #             delta_J[k] = 2 * np.sum(S[i, idx])
        #     print(delta_J)
        #     cluster_assignment[i] = np.argmin(delta_J)
        # with open("./temp_model/kmeans/training/cluster_assignment.npy", "wb") as f:
        #     cluster_assignment = np.load("./temp_model/kmeans/training/cluster_assignment.npy")
        #     np.save(f, cluster_assignment)

        cluster_assignment = np.random.randint(self.K, size=(self.N,))
        
        return cluster_assignment
    
    def _init_clusters(self):
        delta = 0.0
        for k in range(self.K):
            (idx,) = np.where(self.cluster_assignment == k)
            probs_k = self.Q[idx]
            # update cluster centroid
            self.cluster_centroids[k] = np.mean(probs_k, axis=0)
            dist = d_js(probs_k, self.cluster_centroids[k])[:, 0]
            # update delta
            delta = max(delta, np.max(dist))
            # representative sample that is closest to centroid
            idx_sel = np.argmin(dist)
            # only keep one sample in cluster
            self.cluster_assignment[idx] = -1
            self.cluster_assignment[idx[idx_sel]] = k

        # update delta
        self.delta = min(self.delta, 2 * delta)

        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init

    @timeit
    def fit(self, max_iter=1000, tol=1e-9, patience=5):
        best_loss = np.inf
        no_improvement_count = 0
        best_clusters = self.cluster_assignment.copy()
        best_centroids = self.cluster_centroids.copy()
        print('initalize centers...')
        # initialize clusters via approximate solution to upper bound minimization
        self.cluster_assignment = self.initialize_centers(self.S, self.K)

        for i in auto.trange(max_iter, position=0, leave=True, desc='cluster assignment...'):  # pylint: disable=unused-variable
            print('initalize cluster...')
            self._init_clusters()
            # create similarity graph G
            print('update by threshold...')
            if(self.big_data):
                update_by_threshold(self.BigS['training'], self.BigG['training'], self.delta)
                G = None
            else:
                G = 1.0 * (self.S <= self.delta)
            while True:
                print('get candidates...')
                idx_candidates = self._get_candidates(G)
                if len(idx_candidates) == 0:
                    break
                print('get reachable clusters...')
                reachable_clusters = self._get_reachable_clusters(G, idx_candidates)
                self._assign_cluster(idx_candidates, reachable_clusters)
            print('update centroids...')
            self._update_centroids()
            print('calculate loss...')
            loss = self._calculate_loss()
            print(f"loss: {loss}")
            if abs(best_loss - loss) < tol:
                break

            if loss < best_loss:
                best_loss = loss
                best_clusters = self.cluster_assignment.copy()
                best_centroids = self.cluster_centroids.copy()
            else:
                no_improvement_count += 1

            if no_improvement_count > patience:
                break

        self.cluster_assignment = best_clusters
        self.cluster_centroids = best_centroids
        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init
        return self

    def _get_candidates(self, G):
        if not len(self.idx_free):
            return []
        
        if self.big_data:
            reachable = self.BigG['training'].big_any(self.idx_free, self.idx_assigned)
        else:
            reachable = np.any(G[self.idx_assigned][:, self.idx_free], axis=0)
        (idx_candidates,) = np.where(reachable)
        return self.idx_free[idx_candidates]

    def _get_reachable_clusters(self, G, candidates):
        # print(self.idx_assigned)
        # print(self.idx_free)
        # print(candidates)
        # print(self.cluster_assignment[self.idx_assigned])
        # print(self.cluster_assignment.shape)
        if self.big_data:
            reachable = self.BigG['training'].chunk_where_multiple(self.idx_assigned, candidates)
            # reachable = [self.BigG.big_where_i_axis(self.idx_assigned, node) for node in candidates]
        else:
            reachable = [np.where(G[self.idx_assigned][:, node])[0] for node in candidates]
        
        
        reachable_clusters = [np.unique(self.cluster_assignment[self.idx_assigned][node]) for node in reachable]
        return reachable_clusters

    def _assign_cluster(self, candidates, reachable_clusters):
        for node, reachable in zip(candidates, reachable_clusters):
            p = self.Q[node]
            p_clusters = self.cluster_centroids[reachable]
            dist = d_js(p, p_clusters)[0]
            self.cluster_assignment[node] = reachable[np.argmin(dist)]
        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init

    def _update_centroids(self):
        for i in range(self.K):
            (cluster_i,) = np.where(self.cluster_assignment == i)
            self.cluster_centroids[i] = np.mean(self.Q[cluster_i], axis=0)

    def _calculate_loss(self):
        loss = 0
        for i in range(self.K):
            (cluster_i,) = np.where(self.cluster_assignment == i)
            P = self.Q[cluster_i]
            P_c = self.cluster_centroids[i]
            dist = d_js(P, P_c)[:, 0]
            loss += np.sum(dist)
        return loss

    def get_cluster(self, k):
        (cluster_k,) = np.where(self.cluster_assignment == k)
        p = self.cluster_centroids[k]
        return {"samples": cluster_k, "p": p}

    def get_outliers(self):
        return self.idx_free

    def predict(self, S, probs, dataset_cfgs):

        dataset = dataset_cfgs['dataset']
        folder_name = f"./temp_model/kmeans/{dataset}"

        if(os.path.exists(f"{folder_name}/cluster_assignment.npy")):
            cluster_assignment = np.load(f"{folder_name}/cluster_assignment.npy")
            print("load pretrained predict cluster.")
            return cluster_assignment
        

        if self.big_data:
            self._init_BigArray(dataset_cfgs)
            update_by_threshold(self.BigS[dataset], self.BigG[dataset], self.delta)
            
        else:
            affinity = 1.0 * (S <= self.delta)

        cluster_assignment = np.full((len(probs),), -1)
        print("calculating probs...")
        for i, p in auto.tqdm(enumerate(probs), total=len(probs)):
            if self.big_data:
                # print(len(probs), i)
                reachable_clusters = self.cluster_assignment[self.BigG[dataset].get_row_i(i) == 1]
                #reachable_clusters = self.cluster_assignment[self.BigG[dataset].get_col_i(len(self.cluster_assignment), i) == 1]
            else:
                reachable_clusters = self.cluster_assignment[affinity[i] == 1]
            reachable_clusters = np.unique(reachable_clusters)
            if len(reachable_clusters) > 1:
                p_clusters = self.cluster_centroids[reachable_clusters]
                dist = d_js(p, p_clusters)[0]
                cluster_assignment[i] = reachable_clusters[np.argmin(dist)]
            elif len(reachable_clusters) == 1:
                cluster_assignment[i] = reachable_clusters.item()
            else:
                print('An outlier occcurs. No given it to any class.')
                pass

            print(cluster_assignment[i])

        return cluster_assignment
    
    def save(self, filename='./temp_model/kmeans/kmeans.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename='./temp_model/kmeans/kmeans.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)
