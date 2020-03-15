import pandas as pd
import numpy as np
import os
from collections import Counter
import torch
from torch_geometric.data import Data

df = pd.read_csv('dataset/COVID19_open_line_list.csv')


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)