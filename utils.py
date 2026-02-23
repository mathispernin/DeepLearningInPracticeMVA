import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def precompute_features(
    model: nn.Module, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: nn.Module
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    # 1. Set the model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # 2. Temporarily replace the final layer with Identity to extract features
    original_fc = model.fc
    model.fc = nn.Identity()
    
    # 3. Use a DataLoader to process the dataset in batches efficiently
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_features = []
    all_labels = []
    
    # 4. Extract features without computing gradients
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Extract features: outputs shape (batch_size, 512)
            features = model(images)
            
            # Move features to CPU so we don't run out of GPU memory 
            # when storing the entire dataset
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    # 5. Restore the original final layer to the model
    model.fc = original_fc
    
    # 6. Concatenate all batches into single tensors
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 7. Return a TensorDataset wrapping the precomputed data
    return TensorDataset(all_features, all_labels)