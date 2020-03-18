from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from src.graph_loader import make_fc_graph
from pathlib import Path
from src.data_loader import load_pickle
import pickle
from mpl_toolkits.basemap import Basemap as Basemap
import pandas as pd
import matplotlib.lines as mlines

SOURCE_PATH = Path(__file__).parent


def visualize_on_map(data, labels):
	'''
	Credits: Code adapted from https://github.com/tuangauss/DataScienceProjects/blob/master/Python/flights_networkx.py
	Args:
		data (pytorch geometric graph):
		labels (dictionary):

	Returns:
		Displays/saves a plot
	'''
	df = pd.read_csv(SOURCE_PATH / '../dataset/lat_lon_usa.csv')

	plt.figure(figsize=(15, 20))
	m = Basemap(
		projection='merc',
		llcrnrlon=-180,
		llcrnrlat=10,
		urcrnrlon=-50,
		urcrnrlat=70,
		lat_ts=0,
		resolution='l',
		suppress_ticks=True)

	data_graph = to_networkx(data)
	pos = {}
	temp_x, temp_y = [], []
	edge_weights = data.edge_attr.numpy()
	degrees = dict(data_graph.degree)
	edges = data_graph.edges
	strong_nodes = []
	weak_nodes = []
	node_strengths = {}
	weightage = 0.5
	for index, edge in enumerate(edges):
		n1, n2 = edge

		if n1 in node_strengths:
			node_strengths[n1] += edge_weights[index]
		else:
			node_strengths[n1] = edge_weights[index] + weightage*degrees[n1]
		if n2 in node_strengths:
			node_strengths[n2] += edge_weights[index]
		else:
			node_strengths[n2] = edge_weights[index] + weightage*degrees[n2]

	print(degrees)
	print(labels)
	for k, v in node_strengths.items():
		print("State: ", labels[k], " Strength: ", v)
	# The top x nodes get assigned the strong nodes category
	x = 10
	counter = 0
	for k, v in sorted(node_strengths.items(), key=lambda item: item[1], reverse=True):
		if counter < x:
			strong_nodes.append(k)
		else:
			weak_nodes.append(k)

		counter += 1

	# Assign cor-ordinates to plot the graph
	for index, node in enumerate(data_graph.nodes):
		state = labels[node]

		lon = df.loc[df['State'] == state, 'lon'].item()
		lat = df.loc[df['State'] == state, 'lat'].item()
		temp_x.append(lon)
		temp_y.append(lat)
	mx, my = m(temp_x, temp_y)

	for index, val in enumerate(data_graph.nodes):
		pos[val] = (mx[index], my[index])

	nx.draw_networkx_nodes(data_graph, pos=pos, nodelist=strong_nodes, node_size=20, node_color='r', alpha=0.8)
	nx.draw_networkx_nodes(data_graph, pos=pos, nodelist=weak_nodes, node_size=15, node_color='b', alpha=0.6)
	nx.draw_networkx_edges(data_graph, pos=pos, edgelist=data_graph.edges, width=1.0, alpha=0.1, arrows=False, edge_color='g')

	nx.draw_networkx_labels(data_graph, pos=pos, labels=labels, font_size=8)

	m.drawcountries(linewidth=3)
	m.drawstates(linewidth=0.2)
	m.drawcoastlines(linewidth=1)
	m.fillcontinents(alpha=0.3)
	line1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")
	line2 = mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue")
	line3 = mlines.Line2D(range(1), range(1), color="green", marker='', markerfacecolor="green")
	plt.legend((line1, line2, line3), ('Strong nodes', 'Weak nodes', 'Edges'),
			   loc=4, fontsize='xx-large')
	plt.savefig(SOURCE_PATH / '../results/usmap.png')
	plt.show()


def visualize_graph(data, labels, positions=None):
	data_graph = to_networkx(data)
	pos = nx.kamada_kawai_layout(data_graph)
	# labels = data.y[list(data_graph)].cpu().detach().numpy()
	plt.figure(1, figsize=(14, 12))

	nx.draw_networkx_nodes(data_graph, pos=pos, nodelist=data_graph.nodes, node_color='r', alpha=0.8)
	# nx.draw(data_graph, cmap=plt.get_cmap('Set1'), node_size=75, linewidths=6)
	nx.draw_networkx_edges(data_graph, pos=pos, edgelist=data_graph.edges, width=1.0, alpha=0.5)

	nx.draw_networkx_labels(data_graph, pos=pos, labels=labels, font_size=16)
	plt.show()


if __name__ == '__main__':
	metadata, data = make_fc_graph()

	with open(SOURCE_PATH / '../dataset/generated/usa/node_to_ss', 'rb') as file:
		node_to_ss = pickle.load(file)

	# visualize_graph(data, node_to_ss)
	visualize_on_map(data, node_to_ss)
	# Load to node to ss dict


