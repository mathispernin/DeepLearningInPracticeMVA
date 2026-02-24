import torch.nn as nn

class LastLayer(nn.Linear):
    def __init__(self):
        """
        Initializes the new final layer for the ResNet model.
        
        ResNet18 (and our modified ResNet10) feature extractors 
        output a tensor of size 512 before the fully connected layer.
        Because we have a 2-class problem, out_features is set to 2.
        """
        super().__init__(in_features=512, out_features=2)