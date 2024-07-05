import os
import numpy as np
import concurrent.futures

def update_by_threshold(BigArray_to_read, BigArray_to_write, threshold):
    chunk_cnt = 0
    while(1):
        if(os.path.exists(f"{BigArray_to_read.link}/{chunk_cnt}.npy")):
            arr = np.load(f"{BigArray_to_read.link}/{chunk_cnt}.npy")
            arr = 1.0 * (arr <= threshold)
            with open(f"{BigArray_to_write.link}/{chunk_cnt}.npy", "wb") as f:
                np.save(f, arr)

            chunk_cnt += 1
        else:
            break

class BigArray:
    def __init__(self, chunked_rows, link):
        self.chunked_rows = chunked_rows
        self.link = link
        self.threshold = None

        # create folder if not exist
        os.makedirs(self.link, exist_ok=True)

    def _read_Si(self, idx):
        return np.load(f"{self.link}/{idx}.npy")

    def executing_S(self, fn, candidates, indices, multiple_col=False):
        candidates = np.array(candidates)
        raw_candidate = candidates.copy()
        indices = np.array(indices)
        futures = []
        chunk_cnt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while(len(candidates)):
                # print(chunk_cnt)
                # Submitting tasks to the executor
                c = candidates[candidates < self.chunked_rows]
                if(multiple_col):
                    futures.append(executor.submit(fn, chunk_cnt, c, indices, raw_candidate))
                else:
                    futures.append(executor.submit(fn, chunk_cnt, c, indices))
                candidates = candidates[candidates >= self.chunked_rows] - self.chunked_rows
                chunk_cnt += 1
        raw_results = [future.result() for future in futures]
        # Collecting results in order
        if(multiple_col):
            transposed = list(zip(*raw_results))
            results = [np.concatenate(column) for column in transposed]
        else:
            results = np.concatenate(raw_results)
        return results

    def _big_sum(self, idx, candidates, indices):
        Si = self._read_Si(idx)
        if candidates.ndim == 1:
            return np.sum(Si[candidates][:, indices], axis=-1)
        else:  # Handling 2D candidates array
            results = []
            for c in candidates:
                results.append(np.sum(Si[c][:, indices], axis=-1))
            return np.sum(results, axis=0)

    def big_sum(self, candidates, indices, along_axis=True):
        candidates = np.array(candidates)
        indices = np.array(indices)

        if candidates.ndim == 1 or along_axis:
            # Original method for 1D candidates or full=True
            return self.executing_S(self._big_sum, candidates, indices)
        else:
            # Modified method for 2D candidates array
            futures = []
            chunk_cnt = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while len(candidates):
                    c = candidates[:, candidates[0] < self.chunked_rows]
                    futures.append(executor.submit(self._big_sum, chunk_cnt, c, indices))
                    candidates = candidates[:, candidates[0] >= self.chunked_rows] - self.chunked_rows
                    chunk_cnt += 1

            # Collecting and summing results along the first axis
            results = [future.result() for future in futures]
            return np.sum(results, axis=0)

    def _big_any(self, idx, candidates, indices):
        Si = self._read_Si(idx)
        return np.any(Si[candidates][:, indices], axis=-1)

    def big_any(self, candidates, indices):
        return self.executing_S(self._big_any, candidates, indices)
    
    def big_where_i_axis(self, idx_assigned, column):
        results = self.executing_S(self._big_where_i_axis, idx_assigned, column)
        return results

    def _big_where_i_axis(self, chunk_idx, idx_assigned, column):
        Si = self._read_Si(chunk_idx)
        return np.where(Si[idx_assigned][:, column])[0]
    

    def chunk_where_multiple(self, idx_assigned, columns):
        # Executing the chunk-wise operation
        idx_assigned = np.array(idx_assigned)
        results = self.executing_S(self._chunk_where_multiple, idx_assigned, columns, True)

        return results

    def _chunk_where_multiple(self, chunk_idx, idx_assigned, columns, raw_idx_assigned):
        Si = self._read_Si(chunk_idx)
        chunk_idxs = [np.where(Si[idx_assigned][:, col])[0] for col in columns]
        idx_offset = raw_idx_assigned[raw_idx_assigned < (chunk_idx * self.chunked_rows)].shape[0]
        return [arr + idx_offset for arr in chunk_idxs]
    

    def get_row_i(self, idx):
        Si = self._read_Si(idx // self.chunked_rows)
        return Si[idx % self.chunked_rows]
    

    def _get_col_i(self, chunk_idx, _, column):
        Si = self._read_Si(chunk_idx)
        # print(Si[:][column].shape)
        return Si[:][column]

    def get_col_i(self, row_nums, col):
        # print(row_nums)
        results = self.executing_S(self._get_col_i, list(range(row_nums)), col)
        return results
