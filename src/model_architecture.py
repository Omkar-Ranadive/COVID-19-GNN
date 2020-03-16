import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
	def __init__(self, metadata):
		super(Net, self).__init__()
		self.metadata = metadata
		self.conv1 = GCNConv(self.metadata.num_features, 16)
		self.conv2 = GCNConv(16, 4)
		self.linear = torch.nn.Linear(4, 1)

	def forward(self, data):
		x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		# x = F.dropout(x, training=True)
		x = self.conv2(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.linear(x)
		return x