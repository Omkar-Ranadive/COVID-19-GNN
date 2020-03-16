import pickle
import torch
from torch_geometric.data import Data
from itertools import permutations
from geopy import distance
import numpy as np
from src.data_loader import save_pickle, load_pickle
from pathlib import Path

SOURCE_PATH = Path(__file__).parent

class metadata:
	'''
	A class to store information on graphs
	'''
	def __init__(self, num_features):
		self.num_features = num_features


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
	# Flight dict is used for flight distances for each edge. We will consider an undirected graph, i.e s1, s2 == s2, s1
	flightdict = load_pickle(SOURCE_PATH / '../dataset/generated/usa/flightdict')
	flight_counts = []
	for index, edge in enumerate(edges):
		n1, n2 = edge
		k1, k2 = node_to_ll[n1], node_to_ll[n2]
		s1, s2 = ll_dict[k1][-2], ll_dict[k2][-2]
		pair = frozenset([s1, s2])
		if pair in flightdict:
			flight_counts.append(flightdict[pair])
		else:
			flight_counts.append(0)

		# Calculate the distance based on lat and lon
		dist = distance.distance(k1, k2).miles
		distances.append(dist)

		if index % 10000 == 0:
			print("Processed {} edges!".format(index))

	# Sanity check
	assert len(distances) == len(edges), "Num of edges and distances are not the same"

	# Save the distances as it is too costly to compute it every time
	with open(SOURCE_PATH / '../dataset/generated/usa/distances_usa', 'wb') as file:
		pickle.dump(distances, file)

	# Save the distances as it is too costly to compute it every time
	with open(SOURCE_PATH / '../dataset/generated/usa/flight_counts_usa', 'wb') as file:
		pickle.dump(flight_counts, file)

	# Save the edge tuples
	with open(SOURCE_PATH / '../dataset/generated/usa/edges_usa', 'wb') as file:
		pickle.dump(edges, file)

	# Save the two dictionaries too
	with open(SOURCE_PATH / '../dataset/generated/usa/ll_to_node', 'wb') as file:
		pickle.dump(ll_to_node, file)

	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ll', 'wb') as file:
		pickle.dump(node_to_ll, file)


def normalize(data, std_norm=False):
	'''
	Normalizes data between 0 and 1
	Args:
		data (numpy array): 1D numpy array
	Returns:
		Normalized weights
	'''

	if std_norm:
		data_norm = (data - np.mean(data))/np.std(data)
	else:
		max_dist = np.max(data)
		min_dist = np.min(data)

		data_norm = (data - min_dist)/(max_dist - min_dist)

	return data_norm


def find_edge_weights(distances, flight_counts, normalize_data=True):
	'''
	Calculate edge weight as normalized(1/distance) + normalized(flight counts)
	Args:
		distances (list): List of distances between edges
		normalize_data (bool): Flag to turn normalization on/off
		flight_counts (list): List of flight counts between edges

	Returns: Numpy array consisting of edge weights
	'''
	edge_weights = []
	inverse_distances = []

	for dist in distances:
		inverse_distances.append(1/dist)

	if normalize_data:
		# Normalize the distances
		inverse_distances = normalize(data=np.array(inverse_distances))
		flight_counts = normalize(data=np.array(flight_counts))

	# Calculate the combined edge weight
	for i in range(len(inverse_distances)):
		total_weight = inverse_distances[i] + flight_counts[i]
		edge_weights.append(total_weight)

	return np.array(edge_weights)


def create_node_features():
	# Load to node to ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ll', 'rb') as file:
		node_to_ll = pickle.load(file)

	# Load the ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)

	# Load the population info
	with open(SOURCE_PATH / '../dataset/generated/usa/pop_info', 'rb') as file:
		pop_info = pickle.load(file)

	features = []
	for k, v in sorted(node_to_ll.items()):
		state = ll_dict[v][-2]
		features.append(pop_info[state])

	return np.array(features)


def make_fc_graph():
	'''
	Functions assumes that edges, distances are already calculated

	Returns (pytorch.geometric data object): Returns a graph representation based on edges and node values
	'''
	# Load the edges
	with open(SOURCE_PATH / '../dataset/generated/usa/edges_usa', 'rb') as file:
		edges = pickle.load(file)
	# Load the distances
	with open(SOURCE_PATH / '../dataset/generated/usa/distances_usa', 'rb') as file:
		distances = pickle.load(file)
	with open(SOURCE_PATH / '../dataset/generated/usa/flight_counts_usa', 'rb') as file:
		flights = pickle.load(file)
	# Compute the edge weights
	edge_weights = find_edge_weights(distances=distances, flight_counts=flights, normalize_data=True)

	edge_index = torch.tensor(edges, dtype=torch.long)

	# Load to node to ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ll', 'rb') as file:
		node_to_ll = pickle.load(file)

	# Load the ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)


	# Output of the graph network will be number of cases at that node
	y = []
	for k, v in sorted(node_to_ll.items()):
		y.append(ll_dict[v][-1])

	y = np.array(y).reshape(-1, 1)
	node_values = create_node_features()
	x = torch.tensor(node_values, dtype=torch.float)
	y = torch.tensor(y, dtype=torch.float)
	edge_weights = torch.tensor(edge_weights, dtype=torch.float)
	data = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_weights)
	print(data)
	md = metadata(num_features=data.x.shape[1])
	return md, data


if __name__ == '__main__':
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)

	# find_distances(ll_dict)

	make_fc_graph()
	#
	# with open('../dataset/generated/usa/distances_usa', 'rb') as file:
	# 	distances = pickle.load(file)
	#
	# with open('../dataset/generated/usa/flight_counts_usa', 'rb') as file:
	# 	flights = pickle.load(file)
	# # edge_weights = find_edge_weights(distances, flights)
	# make_fc_graph()
	#
	# edge_index = torch.tensor([[0, 1, 1, 2],
	# 						   [1, 0, 2, 1]], dtype=torch.long)
	# x = torch.tensor([[-1, 2], [0, 5], [1, 6]], dtype=torch.float)
	#
	# data = Data(x=x, edge_index=edge_index)
	# print(data['x'])