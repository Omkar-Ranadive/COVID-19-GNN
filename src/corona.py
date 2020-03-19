import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from src.utils import load_pickle

SOURCE_PATH = Path(__file__).parent / '../dataset/timeseries/data'


class Corona(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Corona, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        features = load_pickle(SOURCE_PATH / 'features.pkl')
        dist_matrix = load_pickle(SOURCE_PATH / 'dist_matrix.pkl')
        travel_matrix = load_pickle(SOURCE_PATH / 'travel_matrix.pkl')
        dist_matrix = 1 - dist_matrix
        dist_matrix = dist_matrix/np.max(dist_matrix) + travel_matrix/np.max(travel_matrix)
        dist_matrix = dist_matrix/np.max(dist_matrix)
        dist_matrix[dist_matrix <= 0.2] = 0

        print(dist_matrix)
        # for i in range(len(dist_matrix)):
        #     for j in range(len(dist_matrix)):
        #         if i == j:
        #             dist_matrix[i,j] = 1
        y = torch.Tensor(features[:, 0]).float()


        edge_index = torch.Tensor(np.argwhere(dist_matrix != 0).transpose()).long()
        edge_attr = torch.Tensor(dist_matrix[np.nonzero(dist_matrix)].reshape(edge_index.shape[1], )).long()
        # print(edge_attr.size())
        x = torch.Tensor(features[:, 1:]).float()
        n_features = x.shape[0]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_features=n_features)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



