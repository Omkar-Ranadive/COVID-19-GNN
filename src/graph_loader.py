import pickle
import torch
from torch_geometric.data import Data
from itertools import permutations
from geopy import distance
import numpy as np


def find_distances(ll_dict):
	'''
	Find the distance between two (lat, lon) pairs

	Args:
		ll_dict (dict): A dictionary where keys are (latitude, longitude) pairs
	Returns: None (Saves the distances between edges in datasets/generated folder)
	'''

	# Map the (lat, lon) pair to node values
	ll_to_node = dict()
	node_to_ll = dict()

	for index, k in enumerate(sorted(ll_dict.keys())):
		ll_to_node[k] = index
		node_to_ll[index] = k

	# Now, a fully connected graph will include all 2 tuple combination of nodes
	num_nodes = len(node_to_ll)
	edges = list(permutations(range(num_nodes), 2))

	# Define edge weights
	distances = []
	for index, edge in enumerate(edges):
		n1, n2 = edge
		k1, k2 = node_to_ll[n1], node_to_ll[n2]
		# Calculate the distance based on lat and lon
		dist = distance.distance(k1, k2).miles
		distances.append(dist)

		if index % 10000 == 0:
			print("Processed {} edges!".format(index))

	# Sanity check
	assert len(distances) == len(edges), "Num of edges and distances are not the same"

	# Save the distances as it is too costly to compute it every time
	with open('../dataset/generated/distances', 'wb') as file:
		pickle.dump(distances, file)

	# Save the edge tuples
	with open('../dataset/generated/edges', 'wb') as file:
		pickle.dump(edges, file)

	# Save the two dictionaries too
	with open('../dataset/generated/ll_to_node', 'wb') as file:
		pickle.dump(ll_to_node, file)

	with open('../dataset/generated/node_to_ll', 'wb') as file:
		pickle.dump(node_to_ll, file)


def normalize(data):
	'''
	Normalizes data between 0 and 1
	Args:
		data (numpy array): 1D numpy array
	Returns:
		Normalized weights
	'''

	max_dist = np.max(data)
	min_dist = np.min(data)

	data_norm = (data - min_dist)/(max_dist - min_dist)

	return data_norm


def find_edge_weights(distances, normalize_data=True):
	'''
	Calculate edge weight as 1/distance
	Args:
		distances (list): List of distances between edges
		normalize_data (bool): Flag to turn normalization on/off

	Returns: Numpy array consisting of edge weights
	'''
	edge_weights = []
	for dist in distances:
		edge_weights.append(1/dist)

	if normalize_data:
		# Normalize the distances
		edge_weights = normalize(data=np.array(edge_weights))

	return edge_weights


def make_fc_graph():
	'''
	Functions assumes that edges, distances are already calculated

	Returns (pytorch.geometric data object): Returns a graph representation based on edges and node values
	'''
	# Load the edges
	with open('../dataset/generated/edges', 'rb') as file:
		edges = pickle.load(file)
	# Load the distances
	with open('../dataset/generated/distances', 'rb') as file:
		distances = pickle.load(file)
	# Compute the edge weights
	edge_weights = find_edge_weights(distances=distances, normalize_data=True)

	edge_index = torch.tensor(edges, dtype=torch.long)

	# Node value can be the lat and lon co-ordinates
	with open('../dataset/generated/node_to_ll', 'rb') as file:
		node_to_ll = pickle.load(file)

	# Form a list out of it
	node_values = []
	for k, v in sorted(node_to_ll.items()):
		node_values.append(v)

	node_values = np.array(node_values)

	x = torch.tensor(node_values, dtype=torch.float)
	edge_weights = torch.tensor(edge_weights, dtype=torch.float)
	data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_weights)
	print(data)

	return data


if __name__ == '__main__':
	with open('../dataset/generated/lldict', 'rb') as file:
		ll_dict = pickle.load(file)

	make_fc_graph()
	# with open('../dataset/generated/distances', 'rb') as file:
	# 	distances = pickle.load(file)
	# edge_weights = find_edge_weights(distances)
	#
	# edge_index = torch.tensor([[0, 1, 1, 2],
	# 						   [1, 0, 2, 1]], dtype=torch.long)
	# x = torch.tensor([[-1, 2], [0, 5], [1, 6]], dtype=torch.float)
	#
	# data = Data(x=x, edge_index=edge_index)
	# print(data['x'])