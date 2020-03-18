import torch
import torch.nn.functional as F
from src.model_architecture import Net
from src.graph_loader import make_fc_graph
import matplotlib.pyplot as plt

# Generate the graph
metadata, graph = make_fc_graph()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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