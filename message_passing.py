import torch
import torch_geometric
from torch_geometric.data import Data


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    
    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (remove it for the bonus)
    # Note: These are kept as requested, but the code below is generic enough to work 
    # if they were removed (provided stride remains 1).
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, "Expected padding of 2 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, "Expected kernel size of 5x5."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape
    
    # 1. Setup Kernel parameters
    # If conv2d is provided, we use its params, otherwise default to the assignment assumptions (5x5, pad 2)
    if conv2d is not None:
        k_h, k_w = conv2d.kernel_size
        pad_h, pad_w = conv2d.padding
    else:
        k_h, k_w = 5, 5
        pad_h, pad_w = 2, 2

    # 2. Node Features (x)
    # Permute to (H, W, C) so flattening preserves row-major order (pixel 0 is at 0,0; pixel 1 is at 0,1...)
    x = image.permute(1, 2, 0).view(-1, C)

    # 3. Edges (edge_index) and Attributes (edge_attr)
    # We iterate over every possible relative position in the kernel (e.g., -2 to +2).
    # For each relative position, we find all valid pixel pairs (source -> target).
    
    sources = []
    targets = []
    edge_attrs = []
    
    # Create a grid of node indices corresponding to the image (H, W)
    node_grid = torch.arange(H * W, device=image.device).view(H, W)

    # Iterate over kernel height (dy) and width (dx)
    # dy corresponds to row offset, dx to column offset
    # The kernel covers [-pad, ..., +pad] relative to center
    # We map relative offsets to kernel indices 0..K^2-1
    for dy_idx in range(k_h):
        for dx_idx in range(k_w):
            # Calculate the relative offset (e.g., -2 for index 0 in a 5x5)
            dy = dy_idx - pad_h
            dx = dx_idx - pad_w
            
            # Calculate the kernel weight index for this position (flat index)
            # This will be our edge attribute
            k_idx = dy_idx * k_w + dx_idx

            # Define the range of valid "Center" (Target) pixels (y, x)
            # A center pixel at y is valid for offset dy if its neighbor (y+dy) is inside [0, H).
            # So: 0 <= y < H  AND  0 <= y + dy < H
            # Implies: max(0, -dy) <= y < min(H, H - dy)
            y_start, y_end = max(0, -dy), min(H, H - dy)
            x_start, x_end = max(0, -dx), min(W, W - dx)

            # If the intersection is empty (e.g. image too small for kernel), skip
            if y_start >= y_end or x_start >= x_end:
                continue

            # TARGETS: The pixels currently being updated (the centers)
            target_indices = node_grid[y_start:y_end, x_start:x_end].flatten()

            # SOURCES: The neighbor pixels
            # They are shifted by (dy, dx) relative to the targets
            source_indices = node_grid[y_start+dy : y_end+dy, x_start+dx : x_end+dx].flatten()

            sources.append(source_indices)
            targets.append(target_indices)
            
            # ATTRIBUTES: The kernel index (k_idx) repeated for all these edges
            # We use a 1D tensor of Long type
            attr = torch.full((source_indices.size(0),), k_idx, dtype=torch.long, device=image.device)
            edge_attrs.append(attr)

    # 4. Aggregate results
    if sources:
        # edge_index shape: (2, Num_Edges)
        edge_index = torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)
        # edge_attr shape: (Num_Edges, )
        edge_attr = torch.cat(edge_attrs)
    else:
        # Handle empty case
        edge_index = torch.empty((2, 0), dtype=torch.long, device=image.device)
        edge_attr = torch.empty((0,), dtype=torch.long, device=image.device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, "Expected padding of 2 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, "Expected kernel size of 5x5."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    # 1. Verify consistency
    # data is of shape (Num_Nodes, Channels)
    num_nodes, channels = data.shape
    
    if num_nodes != height * width:
        raise ValueError(
            f"Number of nodes in data ({num_nodes}) does not match "
            f"requested image dimensions ({height}x{width} = {height * width})."
        )

    # 2. Reshape to Spatial Grid
    # We view the data as (H, W, C). 
    # This works because image_to_graph flattened the image in row-major order.
    image_hwc = data.contiguous().view(height, width, channels)

    # 3. Permute to PyTorch format (C, H, W)
    image_chw = image_hwc.permute(2, 0, 1)

    return image_chw

class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # 1. Call parent constructor
        # We use 'add' aggregation because convolution sums the weighted neighbors.
        super().__init__(aggr='add')

        # 2. Store Convolution Parameters
        # The weights in Conv2d are of shape (Out, In, H, W).
        # We need to access them based on the 'edge_attr', which is the kernel index k (0 to 24).
        # We reshape the weights to (H*W, Out, In) so we can index the first dimension with edge_attr.
        weight = conv2d.weight
        # Flatten spatial dims: (Out, In, K*K) -> Permute: (K*K, Out, In)
        self.kernel_weights = weight.flatten(2).permute(2, 0, 1).contiguous()
        
        # Store bias (to handle the bonus point / penalty condition)
        self.bias = conv2d.bias

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        
        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the source node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x 1 or E,).
            Contains the kernel index (0..24) for each edge.

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size E x out_channels)
        """
        # 1. Select the correct weight matrix for each edge
        # self.kernel_weights shape: (K*K, Out, In)
        # edge_attr shape: (E,)
        # specific_weights shape: (E, Out, In)
        specific_weights = self.kernel_weights[edge_attr]

        # 2. Perform Matrix Multiplication: W * x
        # specific_weights: (E, Out, In)
        # x_j: (E, In)
        # We want (E, Out). We can use torch.einsum for clear batch multiplication.
        # 'eoi' = edge, out, in
        # 'ei'  = edge, in
        # -> 'eo' = edge, out
        return torch.einsum('eoi,ei->eo', specific_weights, x_j)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Updates node embeddings after aggregation. 
        This is where the Bias is added (gamma function).
        """
        if self.bias is not None:
            return aggr_out + self.bias
        return aggr_out
