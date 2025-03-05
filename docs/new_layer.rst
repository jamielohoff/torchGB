Adding a New G-Net Type for Compression
=======================================

To add a new g-net type like the GenomicBottleNet to the , we need to define its 
behavior for all the PyTorch layer types. In this example, we'll use matrix 
decomposition as an example of how to create a new g-net type.

First, let's define the ``MatrixDecompositionGNet`` class, which will represent 
our new g-net type:

..  code-block:: python
    :caption: Defining a new g-net type for compression. Here we use a 
    matrix decomposition as an example.

    class LowRankMatrixDecompositionGNet(nn.Module):
        """
        A specialized type of g-net that uses low-rank matrix decomposition to
        compute the parameters of a layer.

        Args:
            rank (int, optional): Rank for the matrix decomposition. Defaults to 32.

        """

        def __init__(self, sizes: Sequence[int], rank: int = 32) -> None:
            super().__init__()
            self.rank = rank

            # Define two trainable parameters
            self.A = nn.Parameter(torch.randn(sizes[1], self.rank))
            self.A.requires_grad = True

            self.B = nn.Parameter(torch.randn(self.rank, sizes[0]))
            self.B.requires_grad = True

        def forward(self, x: Tensor) -> Tensor:
            # Simply multiply the input by these two matrices
            return torch.matmul(A, B)

