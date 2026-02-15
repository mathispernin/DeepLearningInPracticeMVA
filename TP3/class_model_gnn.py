import torch
import torch.nn as nn
import torch_geometric.nn as graphnn

class StudentModel(nn.Module):
    # We hard code the arguments beacuse you want to call the function with no arguments
    def __init__(self, input_size=50, hidden_size=64, output_size=121, heads=4):
        super(StudentModel, self).__init__()

        # TransformerConv supports multi-head attention
        # We used TransformerConv which leverages a similar attention mechanism to standard Graph Attention Networks (GAT) but gives better performance, 
        #allowing us to surpass the maximum F1 score of 70% achieved during our initial GAT trials.
        self.conv1 = graphnn.TransformerConv(input_size, hidden_size, heads=heads)
        self.conv2 = graphnn.TransformerConv(hidden_size * heads, hidden_size, heads=heads)
        # concat=False to average heads at the end
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
