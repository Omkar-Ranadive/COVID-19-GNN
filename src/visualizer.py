from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from src.graph_loader import make_fc_graph


def visualize_graph(data):
	data_graph = to_networkx(data)
	labels = data.y[list(data_graph)].cpu().detach().numpy()
	plt.figure(1, figsize=(14, 12))
	nx.draw(data_graph, cmap=plt.get_cmap('Set1'), node_color=labels, node_size=75, linewidths=6)
	plt.show()


if __name__ == '__main__':
	metadata, data = make_fc_graph()
	visualize_graph(data)

