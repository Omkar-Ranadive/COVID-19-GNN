import torch
import torch.nn.functional as F
from src.model_architecture import Net
from src.graph_loader import make_fc_graph


# Generate the graph
metadata, graph = make_fc_graph()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(metadata).to(device)
data = graph.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
model.train()
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    print(loss)
    loss.backward()
    optimizer.step()