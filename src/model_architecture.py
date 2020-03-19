import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import SAGEConv


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

#
# class SAGEConv(MessagePassing):
# 	def __init__(self, in_channels, out_channels):
# 		super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
# 		self.lin = torch.nn.Linear(in_channels, out_channels)
# 		self.act = torch.nn.ReLU()
# 		self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
# 		self.update_act = torch.nn.ReLU()
#
# 	def forward(self, x, edge_index):
# 		# x has shape [N, in_channels]
# 		# edge_index has shape [2, E]
#
# 		edge_index, _ = remove_self_loops(edge_index)
# 		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
# 		return self.propagate(edge_index.long(), size=(x.size(0), x.size(0)), x=x)
#
# 	def message(self, x_j):
# 		# x_j has shape [E, in_channels]
#
# 		x_j = self.lin(x_j)
# 		x_j = self.act(x_j)
#
# 		return x_j
#
# 	def update(self, aggr_out, x):
# 		# aggr_out has shape [N, out_channels]
#
# 		new_embedding = torch.cat([aggr_out, x], dim=1)
#
# 		new_embedding = self.update_lin(new_embedding)
# 		new_embedding = self.update_act(new_embedding)
#
# 		return new_embedding


class SAGENet(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(SAGENet, self).__init__()
		self.conv1 = SAGEConv(in_channels, 6, normalize=False)
		self.conv2 = SAGEConv(6, out_channels, normalize=False)

	def forward(self, data):
		# print(data.x.type(), data.edge_index.type(), data.edge_attr.type())
		x = self.conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
		x = F.relu(x)
		x = self.conv2(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
		x = F.relu(x)
		return x


class GNN(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(GNN, self).__init__(aggr='add')  # "Max" aggregation.
		self.fc_message = torch.nn.Linear(in_channels, out_channels)
		self.relu = torch.nn.ReLU()
		self.fc_update = torch.nn.Linear(in_channels + out_channels, out_channels, bias=False)
		self.relu2 = torch.nn.ReLU()

	def forward(self, data):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]
		x, edge_index, edge_weight, idx = data.x, data.edge_index, data.edge_attr, data.idx

		edge_index, _ = remove_self_loops(edge_index)
		# print(edge_index.type())
		edge_index, _ = add_self_loops(edge_index.long(), num_nodes=x.size(0))

		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, idx=idx, edge_weight=edge_weight)

	def message(self, x_j, idx_i, idx_j, edge_weight):
		# x_j has shape [E, in_channels]
		weights = edge_weight[idx_i, idx_j]
		x_j = self.fc_message(x_j)
		x_j = self.relu(x_j)
		x_j = x_j * weights.view(weights.size(0), 1)

		return x_j

	def update(self, aggr_out, x):
		# aggr_out has shape [N, out_channels]

		new_embedding = torch.cat([aggr_out, x], dim=1)

		new_embedding = self.fc_update(new_embedding)
		new_embedding = self.relu2(self.relu(new_embedding))

		return new_embedding
