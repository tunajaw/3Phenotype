import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import auto
from .graph_kmeans import GraphKmeans
from Util.utils import batch_d, EPS


class Cluster:
    def __init__(self, x, z, predictor, K, device, user_prefix='') -> None:
        self.x = (x, x)
        self.prob = z
        self.probs = (z, z)
        self.K = K
        self.predictor = predictor
        self.device = device
        self.user_prefix = user_prefix if user_prefix else "default"
        
        self._chunked_rows = None

    def cluster(self):
        self.generate_distance_matrix()

        if(os.path.exists(f"./temp_model/{self.user_prefix}/kmeans/kmeans.pkl")):
            print('load pretrained kmeans')
            self.kmeans = GraphKmeans.load(filename=f"./temp_model/{self.user_prefix}/kmeans/kmeans.pkl")
        else:
            print('construct kmeans obj')
            self.kmeans = GraphKmeans(self.K, None, self.prob, self._chunked_rows, True, self.user_prefix)
            print('fitting kmeans...')
            self.kmeans.fit()
            print('save kmeans obj...')
            self.kmeans.save(filename=f"./temp_model/{self.user_prefix}/kmeans/kmeans.pkl")
        print('get clusters...')
        self.clusters = [self.kmeans.get_cluster(k) for k in range(self.kmeans.K)]
        print(len(self.clusters))

        self.n_clusters = len(self.clusters)
        labels = np.zeros((len(self.prob),))
        for i, cluster in enumerate(self.clusters):
            labels[cluster["samples"]] = i
        
        return labels


    def generate_distance_matrix(self, chunk_size=200000000):
        x_test, x_corpus = self.x
        probs_test, probs_corpus = self.probs
        N, dim_y = probs_corpus.shape
        M, _ = probs_test.shape
        chunked_rows = min(M, chunk_size // N)
        chunked_arr = np.array([[]])
        chunk_cnt = 0

        print(f"x_test shape:{x_test.shape}")
        print(f"probs_corpus shape:{probs_corpus.shape}")

        folder_name = f"./temp_model/{self.user_prefix}/kmeans/training/A"
        print(folder_name)
        os.makedirs(folder_name, exist_ok=True)

        print(f"N:{N}, M:{M}, chunked_rows:{chunked_rows}")

        # relative sample indexes of (I, J) dimension, e.g. if (x2, x3) are same, then I=[2], J=[3]
        # I, J = self._select_relevant_samples(probs_test, probs_corpus, identical)

        # update chunked_rows for KMeans
        self._chunked_rows = chunked_rows

        # skip if alredy has file
        if sum(1 for entry in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, entry))) == np.ceil(M / chunked_rows):
            print(f"already has distance matrix. chunked_rows={chunked_rows}")
            return int(np.ceil(M / chunked_rows))
        
        print(f"should have {np.ceil(M / chunked_rows)} files, having {sum(1 for entry in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, entry)))} already.")
        
        for i in auto.trange(M, mininterval=1000):
            row_dist = self._batch_path_test(x_test[[i]*N], x_corpus[list(range(N))])
            chunked_arr = np.append(chunked_arr, row_dist)
            if (i % chunked_rows) == chunked_rows-1 :
                with open(f"{folder_name}/{chunk_cnt}.npy", 'wb') as f:
                    chunked_arr = np.reshape(chunked_arr, (chunked_rows, N))
                    np.save(f, chunked_arr)
                    chunked_arr = np.array([[]])
                    chunk_cnt += 1

        if(M % chunked_rows):
            with open(f"{folder_name}/{chunk_cnt}.npy", 'wb') as f:
                chunked_arr = np.reshape(chunked_arr, (M % chunked_rows, N))
                np.save(f, chunked_arr)
                chunk_cnt += 1


        return chunk_cnt
    
    def _batch_path_gen(self, x1, x2, num):
        # x1: batch_size x x_dim
        # x2: batch_size x x_dim
        t = np.linspace(0, 1, num=num).reshape((1, -1, 1)).astype(x1.dtype)

        path = (1 - t) * x1[:, np.newaxis, :] + t * x2[:, np.newaxis, :]
        return path

    def _batch_path_test(self, x1, x2, num=50):
        # x1: batch_size x x_dim
        # x2: batch_size x x_dim
        # paths: batch_size x self.test_num x x_dim
        # print(x1.shape, x2.shape)
        
        paths = self._batch_path_gen(x1, x2, num)
        paths = torch.tensor(paths).to(self.device)
        full_valid_mask = torch.ones((paths.shape[0], num, 1)).to(self.device)
        # print(paths.shape, full_valid_mask.shape)
        probs = nn.Sigmoid()(self.predictor(paths, full_valid_mask)).detach().cpu().numpy()
        # print(probs.shape)
        # print(probs)
        
        # print(f"probs:{probs}")
        
        # d_paths: batch_size x self.test_num
        # d_paths1 = batch_KL(probs[:], probs[:, [0]])[:, :, 0]
        # d_paths2 = batch_KL(probs[:, [0]], probs[:, :])[:, 0, :]
        # d_paths = 0.5 * (d_paths1 + d_paths2)
        # batch_size x test_num
        d_paths1 = batch_d(probs, probs[:, [0]])[:, :, 0]
        d_paths2 = batch_d(probs, probs[:, [-1]])[:, :, 0]

        # batch_size x 2*test_num
        d_paths = np.concatenate([d_paths1, d_paths2], axis=-1)
        # equivalent: batch_size
        dist = np.max(d_paths, axis=-1)

        # # not monotory
        # if(d_paths2[0][0] != dist):
        #     print(f"not monotory e:{np.sum((x2-x1)**2)}")
        #     print(f"x1:{d_paths1}")
        #     print(f"x2:{d_paths2}")
        #     print(f"dist:{dist}")
        # else:
        #     if(np.sum((x2-x1)**2) >= 0.1):
        #         print(f"monotory e: {np.sum((x2-x1)**2)}")
            # print(f"monotory e:{np.sum((x2-x1)**2)}")
        # print(dist)

        # assert 0
        return dist