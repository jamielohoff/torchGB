Adding a New G-Net Type for Compression
=======================================

To add a new g-net type like the GenomicBottleNet to the , we need to define its 
behavior for all the PyTorch layer types. In this example, we'll use matrix 
decomposition as an example of how to create a new g-net type.

First, let's define the `MatrixDecompositionGNet` class, which will represent 
our new g-net type:

..  code-block:: python
    :caption: Defining a new g-net type for compression. Here we use a 
    matrix decomposition as an example.
    
class MatrixDecompositionGNet(nn.Module):
    """
    A specialized type of g-net that uses matrix decomposition to parallelize
    the computation of different tiles.

    Args:
        nn (nn.Module): PyTorch neural network module.
    """

    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # TO DO: implement matrix decomposition logic here
        pass

