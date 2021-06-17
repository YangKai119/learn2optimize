import itertools
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from tqdm import tqdm

def tsp_opt(points):
    """
    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    a = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        b = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                b[(S, j)] = min([(a[(S-{j}, k)][0] + all_distances[k][j], a[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        a = b
    res = min([(a[d][0] + all_distances[0][d[1]], a[d][1]) for d in iter(a)])
    return np.asarray(res[1])

class TSPDataset(Dataset):
    """
    Random TSP dataset
    """
    def __init__(self, data_size, min_seq_len, max_seq_len, solver=tsp_opt, solve=True):
        self.data_size = data_size
        self.min_leq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.solve = solve
        self.solver = solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        length = len(self.data['Points_List'][idx])
        solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None
        return tensor, length, solution

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.random((np.random.randint(self.min_leq_len, self.max_seq_len), 2)))
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i' % (i+1, len(points_list)))
                solutions.append(self.solver(points))
        else:
            solutions = None

        return {'Points_List':points_list, 'Solutions':solutions}

    def _to1hot_vec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.max_seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec


def sparse_seq_collate_fn(batch):
    batch_size = len(batch)
    sorted_seqs, sorted_lengths, sorted_label = zip(*sorted(batch, key=lambda x:x[1], reverse=True))
    padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]
    # (sparse) batch_size X max_seq_len X input_dim
    seq_tensor = torch.stack(padded_seqs)
    # batch_size
    length_tensor = torch.LongTensor(sorted_lengths)
    padded_labels = list(zip(*(itertools.zip_longest(*sorted_label, fillvalue=-1))))
    # batch_size X max_seq_len (-1 padding)
    label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)

    # TODO: Currently, PyTorch DataLoader with num_workers >= 1 (multiprocessing) does not support Sparse Tensor
    # TODO: Meanwhile, use a dense tensor when num_workers >= 1.
    # seq_tensor = seq_tensor.to_dense()

    return seq_tensor, length_tensor, label_tensor

