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


def create_state_mappings(ll_dict):
	'''
	A higher granularity solution where each node represents a state instead of a town/city etc.
	Args:
		lldict: Input dictionary with key = (lat, lon)

	Returns:
		A dictionary with key = state value
	'''

	state_dict = dict()

	for k, v in ll_dict.items():
		state = v[-2]
		if state in state_dict:
			state_dict[state][0].append(k)  # This will (lat, lon) coordinates associated with that state
			state_dict[state][1] += v[-1]  # This will keep track of actual cases in that state
		else:
			state_dict[state] = [[k], v[-1]]

	return state_dict


def find_distances_state_based(ll_dict):
	# Map the (lat, lon) pair to node values
	ss_to_node = dict()
	node_to_ss = dict()

	state_dict = create_state_mappings(ll_dict)

	for index, k in enumerate(sorted(state_dict.keys())):
		ss_to_node[k] = index
		node_to_ss[index] = k

	# Now, a fully connected graph will include all 2 tuple combination of nodes
	num_nodes = len(node_to_ss)
	edges = list(permutations(range(num_nodes), 2))

	# Define edge weights
	distances = []
	# Flight dict is used for flight distances for each edge. We will consider an undirected graph, i.e s1, s2 == s2, s1
	flightdict = load_pickle(SOURCE_PATH / '../dataset/generated/usa/flightdict')
	flight_counts = []
	for index, edge in enumerate(edges):
		n1, n2 = edge
		s1, s2 = node_to_ss[n1], node_to_ss[n2]

		pair = frozenset([s1, s2])
		if pair in flightdict:
			flight_counts.append(flightdict[pair])
		else:
			flight_counts.append(0)

		# Calculate the average distance for the state pairs
		c1 = state_dict[s1][0]
		c2 = state_dict[s2][0]
		dist = 0
		for i in c1:
			for j in c2:
				dist += distance.distance(i, j).miles

		avg_dist = dist/(len(c1)*len(c2))
		distances.append(avg_dist)
		if index % 10000 == 0:
			print("Processed {} edges!".format(index))

	# Sanity check
	assert len(distances) == len(edges), "Num of edges and distances are not the same"

	# Save the distances as it is too costly to compute it every time
	with open(SOURCE_PATH / '../dataset/generated/usa/distances_usa_sb', 'wb') as file:
		pickle.dump(distances, file)

	# Save the distances as it is too costly to compute it every time
	with open(SOURCE_PATH / '../dataset/generated/usa/flight_counts_usa_sb', 'wb') as file:
		pickle.dump(flight_counts, file)

	# Save the edge tuples
	with open(SOURCE_PATH / '../dataset/generated/usa/edges_usa_sb', 'wb') as file:
		pickle.dump(edges, file)

	# Save the two dictionaries too
	with open(SOURCE_PATH / '../dataset/generated/usa/ss_to_node', 'wb') as file:
		pickle.dump(ss_to_node, file)

	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ss', 'wb') as file:
		pickle.dump(node_to_ss, file)


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


def create_node_features_states():
	# Load to node to ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ss', 'rb') as file:
		node_to_ss = pickle.load(file)

	# Load the ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)

	# Load the population info
	with open(SOURCE_PATH / '../dataset/generated/usa/pop_info', 'rb') as file:
		pop_info = pickle.load(file)

	features = []
	for k, v in sorted(node_to_ss.items()):
		features.append(pop_info[v])

	features = np.array(features)
	# Normalize each column individually
	columns = features.shape[1]
	for i in range(columns):
		features[:, i] = normalize(features[:, i], std_norm=True)

	return np.array(features)


def filter_edges(edges, edge_weights, threshold):
	'''

	A functio to delete edges from a graph if the edge weights are below a certain threshold
	Args:
		edges (numpy array): An array containing the edge tuples
		edge_weights: An array containing the weights associated with those edges
		threshold (float): A float value determining the minimum edge weight

	Returns:(numpy arrays)
	Filtered edges and edge weights
	'''

	filtered_indices = np.where(edge_weights >= threshold)[0]
	filtered_edges = edges[filtered_indices]
	filtered__weights = edge_weights[filtered_indices]

	return filtered_edges, filtered__weights


def make_fc_graph():
	'''
	Functions assumes that edges, distances are already calculated

	Returns (pytorch.geometric data object): Returns a graph representation based on edges and node values
	'''

	# For lower-granularity

	# For state based
	# Load the edges
	with open(SOURCE_PATH / '../dataset/generated/usa/edges_usa_sb', 'rb') as file:
		edges = pickle.load(file)
	# Load the distances
	with open(SOURCE_PATH / '../dataset/generated/usa/distances_usa_sb', 'rb') as file:
		distances = pickle.load(file)
	with open(SOURCE_PATH / '../dataset/generated/usa/flight_counts_usa_sb', 'rb') as file:
		flights = pickle.load(file)

	# Compute the edge weights
	edge_weights = find_edge_weights(distances=distances, flight_counts=flights, normalize_data=True)

	# Filter the edges if needed
	edges, edge_weights = filter_edges(np.array(edges), edge_weights, threshold=0.1)

	edge_index = torch.tensor(edges, dtype=torch.long)

	# Load to node to ll dict
	# with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ll', 'rb') as file:
	# 	node_to_ll = pickle.load(file)

	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ss', 'rb') as file:
		node_to_ss = pickle.load(file)

	# Load the ll dict
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)

	# Output of the graph network will be number of cases at that node
	y = []
	# for k, v in sorted(node_to_ll.items()):
	# 	# 	y.append(ll_dict[v][-1])
	state_dict = create_state_mappings(ll_dict)

	for k, v in sorted(node_to_ss.items()):
		y.append(state_dict[v][-1])

	# Create training and testing masks
	np.random.seed(0)
	num_nodes = len(y)
	random_order = np.random.permutation(num_nodes)
	train_percent = 0.8
	train_nodes = int(train_percent*num_nodes)
	train_mask, test_mask = np.zeros(num_nodes, dtype=bool), np.zeros(num_nodes, dtype=bool)
	train_mask[random_order[:train_nodes]] = True
	test_mask[random_order[train_nodes:]] = True

	y = np.array(y).reshape(-1, 1)
	# node_values = create_node_features()
	node_values = create_node_features_states()
	x = torch.tensor(node_values, dtype=torch.float)
	y = torch.tensor(y, dtype=torch.float)
	train_mask = torch.tensor(train_mask, dtype=torch.bool)
	test_mask = torch.tensor(test_mask, dtype=torch.bool)
	edge_weights = torch.tensor(edge_weights, dtype=torch.float)
	data = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_weights, train_mask=train_mask,
				test_mask=test_mask)
	print(data)
	md = metadata(num_features=data.x.shape[1])
	return md, data


if __name__ == '__main__':
	with open(SOURCE_PATH / '../dataset/generated/usa/lldict_usa', 'rb') as file:
		ll_dict = pickle.load(file)

	# find_distances_state_based(ll_dict)
	# find_distances(ll_dict)
	make_fc_graph()
	#

