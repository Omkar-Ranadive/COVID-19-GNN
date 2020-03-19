import torch
import torch.nn.functional as F
from src.model_architecture import Net, SAGENet, GNN
from src.graph_loader import make_fc_graph
import matplotlib.pyplot as plt
from src.corona import Corona
from src.utils import save_pickle, load_pickle
from pathlib import Path
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import os

SOURCE_PATH = Path(__file__).parent
# Make the models folder if it doesn't exists
if not os.path.exists(SOURCE_PATH /'models'):
    os.mkdir(SOURCE_PATH / 'models')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Enter 1, 2 or 3.")
print("1. Train using simple GCN Architecture: (Non-timeseries data): ")
print("2. Train using Sage Conv Architecture (timeseries data): ")
print("3. Train using Message Passing Architecture (timeseries data): ")
reply = int(input())

if reply == 1:
    # Generate the graph
    metadata, graph = make_fc_graph()
    model = Net(metadata).to(device)
    data = graph.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    num_epochs = 50
    print_freq = 1000
    save_freq = 49
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
        train_losses.append(loss)
        test_losses.append(test_loss)
        if epoch % print_freq == 0:
            print("Epoch: {}, Train Loss: {}, Test Loss: {}".format(epoch, loss, test_loss))
        if epoch % save_freq == 0:
            print("Model saved at epoch {}".format(epoch))
            torch.save(model.state_dict(), 'models/m1_basic_epoch{}.pt'.format(epoch))

        loss.backward()
        optimizer.step()


    plt.plot(range(num_epochs), train_losses, color='r', label='Train Loss')
    plt.plot(range(num_epochs), test_losses, color='b', label='Test Loss')
    plt.legend()
    plt.show()

elif reply == 2:
    num_epochs = 100
    transformed_dataset = Corona(root='')
    train_loader = DataLoader(transformed_dataset)
    model = SAGENet(14, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    cost = torch.nn.MSELoss()

    # print(train_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            # print(data.x.size())
            optimizer.zero_grad()
            output = model(data)
            label = data.y.to(device).reshape(-1, 1)
            loss = cost(output, label)
            loss.backward()
            total_loss = loss
            optimizer.step()

    # for actual, predicted in zip(label, output):
    #     print(actual, predicted)
    # print("actual = ", label, "predicted = ", output, "loss =", total_loss)

    save_pickle(output.cpu().detach().numpy().reshape(-1, ), SOURCE_PATH / 'dataset/timeseries/data/outputs.pkl')

elif reply == 3:
    epochs = 12
    lr = 0.01
    cases = load_pickle(SOURCE_PATH / 'dataset/timeseries/data/features.pkl')
    distances = load_pickle(SOURCE_PATH / 'dataset/timeseries/data/dist_matrix.pkl')
    n_flights = load_pickle(SOURCE_PATH / 'dataset/timeseries/data/travel_matrix.pkl')
    distances = 1 - distances

    norm = np.max(cases)
    cases = cases / np.max(cases)
    edges = distances / np.max(distances) + n_flights / np.max(n_flights)
    edges /= np.max(edges)

    labels = torch.FloatTensor(cases[:, 0])
    features = torch.FloatTensor(cases[:, 1:])
    edge_attr = torch.FloatTensor(edges[np.nonzero(edges)].reshape(-1, 1))
    edge_index = torch.FloatTensor(np.argwhere(edges != 0).transpose())
    # edges[edges <= 0.2] = 0
    edges = torch.FloatTensor(edges)
    idx = torch.tensor(list(range(features.size(0))))
    data = Data(x=features, edge_index=edge_index, edge_attr=edges, y=labels, idx=idx)
    model = GNN(data.x.size(1), 1).to(device)
    model.train()
    data = data.to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        # print(pred.size())
        cost = loss(pred, data.y)
        print(cost)
        cost.backward()
        optimizer.step()
    for x, y in zip(pred, data.y):
        print(int(x.item() * norm), int(y.item() * norm))

else:
    raise ValueError("Invalid response")

