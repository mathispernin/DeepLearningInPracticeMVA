import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader, TensorDataset
def precompute_features(
    model: nn.Module, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:

    # Set the model to evaluation mode
    model.eval()
    model = model.to(device)

    # Temporarily replace the final layer with Identity to extract features
    original_fc = model.fc
    model.fc = nn.Identity()

    # DataLoader to process the dataset (pq 64?)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_features = []
    all_labels = []

    # Extract features without computing gradients !
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Extract features: outputs shape (batch_size, 512)
            features = model(images)

            # Move features to CPU so we don't run out of GPU memory
            # when storing the entire dataset
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    # Restore the original final layer to the model
    model.fc = original_fc

    # Concatenate all batches into single tensors
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Return a TensorDataset wrapping the precomputed data
    return TensorDataset(all_features, all_labels)