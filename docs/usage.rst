.. _usage:


Using the genomic bottleneck to compress models
===============================================
This guide explains the internals of **torchGB**.

The `torch.distributed` package is used in **torchGB** to parallelize the 
training process across multiple GPUs. This allows for efficient use of 
hardware resources and can significantly speed up training times.

To enable distributed training, you need to initialize the multiprocessing 
environment using `dist.init_process_group()`. You can choose from various 
backends such as `nccl`, `gloo`, or `mpi`.

After initialization, you can create a DistributedDataParallel (DDP) wrapper 
around your model using `DistributedDataParallel(model, device_ids=[rank], output_device=rank)`.

Here's an example code snippet that demonstrates how to initialize the 
distributed environment and wrap your model with DDP:

..  code-block:: python
    :caption: Initializing distributed environment and wrapping model with DDP

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = GPT(**experiment_config["model"]).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

The genomic bottleneck in **torchGB** is responsible for parallelizing the 
training of g-nets across multiple GPUs. To achieve this, we use 
the `predict_weights()` method of the GenomicBottleneck class to compute the weights of each g-net.
The genomic bottleneck in **torchGB** is responsible for parallelizing the
training of g-nets across multiple GPUs. To achieve this, we use
the `predict_weights()` method of the GenomicBottleneck class to compute the 
weights of each g-net.  This method implicitly updates the model weights with 
the g-net predictions.

Here's an example code snippet that demonstrates how to call `predict_weights()`
and propagate gradients through the g-nets:

..  code-block:: python
    :caption: Calling predict_weights() and propagating gradients

    gnets.zero_grad() # Zero g-net gradients
    gnets.predict_weights() # Compute p-net weights using g-nets

    loss.backward() # Backpropagate through p-net
    gnets.backward() # Backpropagate through g-nets

    optimizer.step() # Update p-net parameters
    gnets.step() # Update g-net parameters


How g-nets are distributed across MPI ranks:

The distribution of g-nets is handled internally by the `GenomicBottleneck` class.  
During initialization, the `GenomicBottleneck` class (see `src/torchGB/core.py`) 
analyzes the provided model and creates a set of g-nets for specific layers within 
the model. This mapping of g-nets to layers is stored in the  `gnetdict` 
attribute of the `GenomicBottleneck` class.  Each entry in the `gnetdict` 
corresponds to a layer's parameters and is associated with a `GNetLayer` object 
(also in `src/torchGB/core.py`). The `GNetLayer` object stores important 
information, including the MPI rank (`rank` attribute) where the g-net for that 
layer resides.

The `GenomicBottleneck` class uses `torch.distributed` to manage the 
distribution of g-nets across different ranks. When methods like `zero_grad()`, 
`backward()`, and `step()` are called on the `GenomicBottleneck` instance, they 
internally check the `rank` attribute of each `GNetLayer` in the `gnetdict`. 
Operations are performed only on the g-nets residing on the current MPI rank. 
You can see examples of this logic in the implementations of `zero_grad()`, 
`get_num_params_gnet()`, `step()`, and `load()` within `src/torchGB/core.py`.

The `register_gnet_type` function in `src/torchGB/core.py` is used to associate 
specific layer types (e.g., `nn.Linear`, `nn.Conv2d`) with initialization and 
build functions for their corresponding g-nets. This mechanism allows the 
`GenomicBottleneck` class to create appropriate g-nets for different types of 
layers in the model.

For more details on the implementation, refer to the source code, specifically 
`src/torchGB/core.py`,  `src/torchGB/gnet.py`, and layer-specific files under 
`src/torchGB/layers`. The docstrings and comments within these files provide 
further insights into the internal workings of g-net distribution and management.  
The `__repr__` method in `src/torchGB/core.py` also offers a way to print information 
about the created g-nets and their associated parameters.

To use `torch.distributed` with **torchrun**, you need to launch your training 
script using the `--nproc_per_node` argument. This will enable distributed 
training across multiple GPUs.

Here's an example code snippet that demonstrates how to launch a training script with **torchrun**:

..  code-block:: bash
    :caption: Launching training script with torchrun

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm_gnet_small.py \
    --gpus 1,2,3,4 --seed 42 --language en --batchsize 36 \
    --name test --no_commit --log_level DEBUG

By following these steps and using the provided code snippets, you can efficiently parallelize your training process with **torchGB**.


Tiling of large weight matrices
===============================

The tiling/slicing of large weight matrices is implemented for different PyTorch
layer types in the `src/torchGB/layers` directory. Specifically, the
`conv_gnet.py`, `attn_gnet.py`, and `linear_gnet.py` files contain the implementation.

In all files, the `build_<layer-type>_gnet_output`functions are used to compute
the output of the g-net for each layer type. These functions take the following inputs:

*   `name`: The type of layer (e.g., "conv2d" or "linear")
*   `param`: The weights and bias of the original layer
*   `weights`: The weights of the corresponding g-net
*   `tile_shape`: A tuple specifying the tile size for each dimension

Inside these functions, the following steps are performed:

1.  **Compute tile dimensions**: The number of tiles in each dimension is
computed using the ceiling function (`math.ceil`) to ensure that the entire
weight matrix is covered.

2.  **Rebuild the weight matrix**: For convolutional layers, the
`build_4d_kernel` function is used to reshape the weights into a 4D tensor with
the specified tile shape. The resulting tensor is then cut to match the original
layer's output shape using the `cut_matrix` function.
For attention layers, we use the `tile_matrix` function from the 
`src/torchGB/utils.py` file to tile the weight matrix along its rows. 
Specifically, given a 3x1 tiling (i.e., `row_size=3`, `col_size=1`), the input 
weight matrix is reshaped into tiles of size 3x1, and then swapped to have shape 
(n, 3, 1). The resulting tensor has shape (n, 3, 1) where n is the number of 
columns in the original weight matrix.
For example, if we have a 12x8 weight matrix, the `tile_matrix` function would 
split it into 4 tiles of size 3x1 along its rows:

3.  **Return the sliced g-net weights**: The sliced g-net weights are returned
as the final result of the computation.

Here's an excerpt from the `conv_gnet.py` file showing this implementation:
..  code-block:: python
    :caption: How the convolutional g-net output is built
def build_conv2d_gnet_output(name: str, param: Tensor, weights: Tensor, tile_shape) -> Tensor:
    num_row_tiles = math.ceil(param.shape[0]/tile_shape[0])
    num_col_tiles = math.ceil(param.shape[1]/tile_shape[1])

    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1], param.shape[2], param.shape[3])

    new_weights = build_4d_kernel(weights, shape)
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights
    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1], param.shape[2], param.shape[3])
    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1])

    new_weights = build_4d_kernel(weights, shape)
    new_weights = tile_matrix(weights, tile_shape[0], tile_shape[1])
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights

