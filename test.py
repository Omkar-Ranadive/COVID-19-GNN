import torch
import torch.nn.functional as F
from src.model_architecture import Net
from src.graph_loader import make_fc_graph
from src.data_loader import load_pickle
MODEL_PATH = 'models/m1_basic_epoch500.pt'
lldict = load_pickle('dataset/generated/usa/lldict_usa')
node_to_ss = load_pickle('dataset/generated/usa/node_to_ss')

# Generate the graph
metadata, graph = make_fc_graph()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(metadata).to(device)
data = graph.to(device)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
out = model(data)
loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
print("MSE loss: ", loss)
print("Comparision between predicted and actual: ")
pred = out.cpu().detach().numpy()
actual = data.y.cpu().detach().numpy()
test_mask = data.test_mask.cpu().detach().numpy()

for index, val in enumerate(test_mask):
	if val:
		coordinates = node_to_ss[index]
		print("Node info: ", coordinates)
		print("Predicted: ", pred[index], " Actual: ", actual[index])
		print("------------")