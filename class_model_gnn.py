import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graphnn

# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, heads=4):
        super(StudentModel, self).__init__()

        # TransformerConv supports multi-head attention
        # output dim of first layer = hidden_size * heads
        self.conv1 = graphnn.TransformerConv(input_size, hidden_size, heads=heads)

        # Second layer: input must match previous output dim
        self.conv2 = graphnn.TransformerConv(hidden_size * heads, hidden_size, heads=heads)

        # Third layer: project to output size (concat=False to average heads at the end)
        self.conv3 = graphnn.TransformerConv(hidden_size * heads, output_size, heads=1, concat=False)

        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.elu(x)

        x = self.conv2(x, edge_index)
        x = self.elu(x)

        x = self.conv3(x, edge_index)
        return x


# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
