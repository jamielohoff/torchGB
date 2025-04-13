import time
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor

# Terrible style...@JLo fix this!
from .layers.gnet import *
from .layers.matrix_decomposition import *
from .layers.xox import *

from .utils import get_gnet_batchsize, top_k_values


# Stores how the compression is intended to work for different layer types
gnet_types = {}


# TODO: write this as an abstract interface class
class HyperNetwork(nn.Module):
    def __init__() -> None:
        super().__init__()
    def forward(self) -> Tensor:
        pass


@dataclass
class GNetType:
    """
    Simple container data structure to store the init and build functions
    of a g-net.

    Args:
        name (str): The type of g-net. This should be a unique identifier for each layer type.
        init (Callable): A function that initializes the weights and bias of the layer.
        build (Callable): A function that builds the output structure of the layer.
    """
    name: str
    init: Callable
    build: Callable
    

def register_gnet_type(module: nn.Module, init: Callable[[nn.Module], None], 
                       build: Callable[[nn.Module], None]) -> None:
    """
    This function registers a new g-net type to be used in the Genomic Bottleneck.

    Args:
        module (nn.Module): The type of module for which to register the g-net.
        init (Callable[[nn.Module], None]): A function that initializes the weights and bias of the layer.
        build (Callable[[nn.Module], None]): A function that builds the output structure of the layer.
    """
    global gnet_types
    gnet_types[module] = GNetType(str(module), init, build)


# TODO: can we make this prettier?
def register(hypernet_type: str) -> None:
    if hypernet_type == "g-net" or hypernet_type == "gnet":
        init_attn_gnet = partial(init_attn_gnet_fast, FastGenomicBottleNet)
        register_gnet_type(nn.TransformerEncoder, init_attn_gnet, build_attn_gnet_output_fast)
        init_conv2d_gnet = partial(init_conv2d_gnet_fast, FastGenomicBottleNet)
        register_gnet_type(nn.Conv2d, init_conv2d_gnet, build_conv2d_gnet_output_fast)
        init_linear_gnet = partial(init_linear_gnet_fast, FastGenomicBottleNet)
        register_gnet_type(nn.Linear, init_linear_gnet, build_linear_gnet_output_fast)
        
    elif hypernet_type == "stochastic g-net" or hypernet_type == "stochastic-gnet":
        init_attn_gnet = partial(init_attn_gnet_fast, FastStochasticGenomicBottleNet)
        register_gnet_type(nn.TransformerEncoder, init_attn_gnet, build_attn_gnet_output_fast)
        init_conv2d_gnet = partial(init_conv2d_gnet_fast, FastStochasticGenomicBottleNet)
        register_gnet_type(nn.Conv2d, init_conv2d_gnet, build_conv2d_gnet_output_fast)
        init_linear_gnet = partial(init_linear_gnet_fast, FastStochasticGenomicBottleNet)
        register_gnet_type(nn.Linear, init_linear_gnet, build_linear_gnet_output_fast)
        
    else:
        raise ValueError(f"HyperNetwork type {hypernet_type} not recognized.")
    

@dataclass
class GNetLayer:
    """
    This class stores all the information about the g-net for a specific layer.
    
    Args:
        name (str): Name of the layer parameter predicted by the g-net.
        rank (int): Rank of the device where the g-net is stored.
        tile_shape (Optional[Sequence[int, int]]): The shape of the tiles used 
            to predict the weights of the layer.
        gnets (Sequence[GenomicBottleNet]): Sequence of g-net models. 
            This is typically a list of MLPs.
        optimizers (Sequence[optim.Optimizer]): The optimizers used to train 
            all the the g-nets.
        gnet_input (Tensor): The input to the g-net. This is a 
            constant tensor that is used to predict the new weights of the 
            layer. They encode the (i,j)-position of every weight in the 
            parameter matrix of the layer.
        weights (Tensor): The weights predicted by the g-net.
        grad_scale (float): The scaling factor for the gradients.   
    """
    name: str
    rank: int
    tile_shape: Optional[Sequence[int]] = None
    gnets: Optional[Sequence[HyperNetwork]] = None
    optimizers: Optional[Sequence[optim.Optimizer]] = None
    schedulers: Optional[optim.lr_scheduler.LRScheduler] = None
    gnet_input: Optional[Sequence[Tensor]] = None
    weights: Optional[Tensor] = None
    grad_scale: Optional[float] = None


class FastGenomicBottleneck(nn.Module):
    """
    The GenomicBottleneck class implements a hypernetwork that predicts all
    learnable weights matrices in a given neural network model. For every weight,
    a g-net is created that predicts the new weights of the layer.
    The input of every g-net is the encoding of the position of a single weight, 
    e.g. i,j in a matrix. These positions can be encoded in a binary fashion, i.e.
    the weigth at position (8, 3) is encoded as (1000,0011). Other encoding types
    are possible such as a one-hot encoding, but these scale badly.
    These encodings of a single weight are then the input for the hypernetwork.
    In case of an MLP, we have one input neuron per bit, i.e. in the above example
    we have 8 input neurons and a single output neuron which predicts the weight.
    Thus the g-net predicts the value of a single weight of the matrix, but we 
    can parallelize the process by batching across the all the weights and
    reshaping the resulting output into the weight tensor.
    When launched with the torchrun --nproc_per_node='num_gpus' command, 
    every g-net is stored on a different device to parallelize the computation. 
    Furthermore, every g-net has its own optimizer.
    Gradients are backpropagated by first backpropagating the gradients through
    the model and then using them as seeds for further backpropagation through
    the g-nets.

    Args:
        model (nn.Module): The neural network model.
        hidden_dim (int): The size of the hidden layers in the g-nets.
        lr (float): The learning rate of the g-nets.
        gnet_batchsize (int): The number of parameters per tile.
        ignore_layers (Optional[Sequence[str]]): A list of layer names and 
            types that should not be predicted using a g-net.
    """
    lr: float
    num_batches: int
    model: nn.Module
    gnetdict: Dict[str, GNetLayer]
    max_tiles_batchsize: int
    
    def __init__(self, model: nn.Module, local_rank_dict: Dict[int, int], 
                 hidden_dim: int = 32, 
                 lr: float = 0.001, gnet_batchsize: int = 10_000, 
                 compression: Optional[float] = None,
                 ignore_layers: Sequence[str] = [], 
                 hypernet_type: str = "g-net",
                 max_tiles_batchsize: int = 8,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.max_tiles_batchsize = max_tiles_batchsize
        self.scheduler = scheduler
        self.local_rank_dict = local_rank_dict
            
        if compression:
            gnet_batchsize = get_gnet_batchsize(compression, hidden_dim, output_dim=2)
            print(f"Set compression of {compression}x, "
                  f"thus gnet_batchsize set to {gnet_batchsize}.")
        
        register(hypernet_type)
        
        # Stores all the information about the gnets
        self.gnetdict = {}
        initialized = set()
        load_per_rank = np.zeros(dist.get_world_size()) 
           
        # Iterate over all the modules in the model
        for name, module in self.model.module.named_modules(): 
            for pname, param in module.named_parameters():    
                _name = name + "." + pname 
                ignore_param = any([lname in _name for lname in ignore_layers])
                if param.requires_grad and not ignore_param \
                    and not "bias" in pname and not _name in initialized:
                    gnet_type = gnet_types.get(type(module))
                    if gnet_type:
                        # This implements a rudimentary load balancer across devices
                        # that removes the bias towards the first device
                        global_rank = np.where(load_per_rank == load_per_rank.min())[0][-1]
                        load_per_rank[global_rank] += param.data.numel()
                    
                        # Here we initialize the g-net for specific layer types
                        if global_rank == dist.get_rank():
                            out_fn = gnet_type.init
                            out = out_fn(pname, param, hidden_dim, gnet_batchsize, max_tiles_batchsize)
                            if out_fn:
                                self._add_gnets(_name, global_rank, param, *out)
                        else:
                            self.gnetdict[_name] = GNetLayer(name=_name, rank=global_rank)
                        initialized.add(_name)
                    
    def __repr__(self) -> str:
        output_str = f"g-net parameters:\n"
        for name in self.gnetdict.keys():
            gnets = self.gnetdict[name]
            output_str += f"Parameter={name}\n" \
                          f"Parameter shape={gnets.weights.shape}\n" \
                          f"g-net input shape={gnets.gnet_input.shape}\n\n"
        return output_str
                                                                         
    def get_num_params_gnet(self) -> int:
        """
        Because gnets are now stored decentralized across devices, we need to
        compute them separately and then sum them up with a all_reduce operation.
        
        Returns:
            int: Cumulative number of parameters of all g-nets.
        """
        rank = self.local_rank_dict[dist.get_rank()]
        num_params = torch.tensor(0).to(rank) 
        
        for name in self.gnetdict.keys():
            n = 0
            if self.gnetdict[name].gnets is not None:
                for gnet in self.gnetdict[name].gnets:
                    n += sum(param.numel() for _, param in gnet.named_parameters())
            num_params += torch.tensor(n).to(rank)
        
        dist.all_reduce(num_params, op=dist.ReduceOp.SUM)
        return num_params.item()
    
    def get_num_params_no_gnet(self) -> int:
        """
        Because gnets are now stored decentralized across devices, we need to
        compute them separately and then sum them up with a all_reduce operation.
        
        Returns:
            int: Cumulative number of parameters of which have no g-nets attached.
        """
        return sum([param.numel() for pname, param in self.model.named_parameters()
                    if not pname in self.gnetdict.keys()])
    
    def compression(self) -> float:
        """
        This function computes the compression ratio of the g-net to the model.

        Returns:
            float: Compression factor of the g-net with respect to the model.
        """
        num_model_params = sum(p.numel() for p in self.model.parameters())
        num_gnet_params = self.get_num_params_gnet() 
        num_gnet_params += self.get_num_params_no_gnet()
        
        compression = num_model_params / num_gnet_params
        return compression
                    
    def save(self, fname: str) -> None:
        """
        Save the g-nets from a checkpoint file.

        Args:
            fname (str): File to which we wish to write the weights of the g-nets.
        """
        checkpoint = {}

        for rank in range(dist.get_world_size()):
            dist.barrier()
            if rank > 0 and dist.get_rank() == rank:
                checkpoint = torch.load(fname, map_location=torch.device("cpu"))
            dist.barrier()
            for name in self.gnetdict.keys():
                if self.gnetdict[name].rank == dist.get_rank() and dist.get_rank() == rank:
                    entry_name = name + "_state_dict"
                    model_name = "model_" + entry_name
                    optimizer_name = "optimizer_" + entry_name
                    
                    checkpoint[model_name] = []
                    checkpoint[optimizer_name] = []
                    d = self.gnetdict[name]
                    
                    for gnet, opt in zip(d.gnets, d.optimizers):
                        checkpoint[model_name].append(gnet.state_dict())
                        checkpoint[optimizer_name].append(opt.state_dict())
            if dist.get_rank() == rank:
                torch.save(checkpoint, fname)
            else:
                time.sleep(1)
            dist.barrier()
                      
    def load(self, fname: str) -> None:
        """
        Loads the g-nets from a specified file.
        This function loads the state dictionaries of the g-nets and their
        corresponding optimizers from a checkpoint file. It ensures that only
        the g-nets corresponding to the current process rank are loaded.

        Args:
            fname (str): File from which to load the g-nets.
        """
        checkpoint = torch.load(fname, map_location=torch.device("cpu"))

        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                entry_name = name + "_state_dict"
                model_name = "model_" + entry_name
                optimizer_name = "optimizer_" + entry_name
                d = self.gnetdict[name]

                for gnet, opt, gnet_params, opt_state in zip(d.gnets, d.optimizers,
                                                             checkpoint[model_name],
                                                             checkpoint[optimizer_name]):
                    gnet.load_state_dict(gnet_params)
                    opt.load_state_dict(opt_state)
                    # TODO: Also load scheduler state

    def train(self) -> None:
        """
        Trains the neural networks in the gnetdict attribute that are assigned 
        to the current process rank. This method iterates over the keys in the 
        gnetdict attribute and checks if the rank of each gnetdict entry
        matches the current process rank. If it does, it sets the corresponding 
        neural networks to training mode.
        """
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for gnet in self.gnetdict[name].gnets:
                    gnet.train()
    
    def zero_grad(self) -> None:
        """
        Zeros out all gradients in the optimizers, similarly to loss.zero_grad().
        This function iterates over all the networks in the gnetdict dictionary.
        For each network that is assigned to the current process rank, it sets the 
        gradients of all associated optimizers to zero.
        """
        
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for optimizer in self.gnetdict[name].optimizers:
                    optimizer.zero_grad()
        
    def predict_weights(self) -> None:
        """
        This function generates the new weights using the gnets, reshapes them
        to the original shape and sets the models parameters to the corresponding
        new weights.
        """
        predicted = set()
        param_list = {i:[] for i in range(dist.get_world_size())}
        for name, mod in self.model.module.named_modules(): 
            for pname, param in mod.named_parameters():    
                _name = name + "." + pname 
                gnetstack = self.gnetdict.get(_name) if not _name in predicted else None
                if gnetstack:
                    if gnetstack.rank == dist.get_rank():
                        new_weights = []
                        tile_shape = gnetstack.tile_shape
                        for gnet in gnetstack.gnets:
                            # This predicts the new weight tiles using the g-nets:
                            gnet_input = gnetstack.gnet_input
                            tiles_batchsize = gnet.num_tiles
                            # ouptputs `tiles_batchsize` many tiles that need to be reshaped
                            gnet_input = gnet_input.unsqueeze(1)
                            new_weight_tiles = gnet(gnet_input)
                            _shape = (tiles_batchsize,) + tile_shape
                            new_weight_tiles = new_weight_tiles.reshape(_shape)
                            new_weights.append(new_weight_tiles)
                    
                        # Assemble the new weight tiles into the full weight matrix
                        new_weights = torch.concatenate(new_weights, dim=0)
                        
                        # Here we build the weight matrix from the tiles 
                        # predicted by the g-nets 
                        gnet_type = gnet_types.get(type(mod))
                        if gnet_type:
                            weights_fn = gnet_type.build
                            new_weights = weights_fn(_name, param, new_weights, tile_shape)
                            
                            # When cutting the matrix, it's not contiguous anymore,
                            # but we need it to be contiguous for broadcasting
                            # across devices
                            new_weights = new_weights.contiguous() 
                            self.gnetdict[_name].weights = new_weights
                            
                            # Sets the parameters to the corresponding new weights
                            param.data = nn.Parameter(new_weights)
                        
                    param_list[self.gnetdict[_name].rank].append(param.data)
                predicted.add(_name)
                
        for source_id in range(dist.get_world_size()):
            for j in range(len(param_list[source_id])):
                # Broadcast the weights of the g-nets calculated on GPU with
                # rank dist.get_rank() to all other GPUs.
                dist.broadcast(param_list[source_id][j], src=source_id)
    
    def backward(self) -> None:
        """
        This function takes the models gradients after a forward and 
        backward pass through the model and propagates them through the g-net to
        update the parameters.
        """           
        backpropagated = set()
        for name, mod in self.model.module.named_modules(): 
            for pname, param in mod.named_parameters():  
                _name = name + "." + pname 
                if not _name in backpropagated:
                    gnetstack = self.gnetdict.get(_name)
                    if gnetstack:
                        if gnetstack.rank == dist.get_rank():
                            gnet_batch_size = gnetstack.gnet_input.size()[0]
                            out_scale = gnetstack.gnets[0].output_scale
                            grad_scale =  gnet_batch_size * torch.square(out_scale)
                            
                            norm_grad = param.grad / grad_scale
                            gnetstack.weights.backward(norm_grad)
                            backpropagated.add(_name)
                      
    def step(self) -> None:
        """
        Performs a single optimization step for each optimizer in the network 
        dictionary.

        This function iterates over the keys in the gnetdict attribute, 
        which is a dictionary containing network objects. For each network 
        object that matches the current process rank, it performs an 
        optimization step using the optimizers associated with that network.

        NOTE: The function currently only steps the optimizers and not the 
        schedulers, as the scheduler step code is commented out.

        Attributes:
            gnetdict (dict): A dictionary where keys are network names and values 
                are network objects that contain optimizers and schedulers.
        """
        for name in self.gnetdict.keys():
            gnetstack = self.gnetdict[name]
            if gnetstack.rank == dist.get_rank():
                iter = zip(gnetstack.optimizers, gnetstack.schedulers)
                for optimizer, scheduler in iter:
                    optimizer.step()
                    scheduler.step()
                    
    def _add_gnets(self, name: str, global_rank: int, param: Tensor,
                   row_col_encodings: Tensor, gnets: HyperNetwork, 
                   tile_shape: Tuple[int, int], output_scale: float,
                   grad_scale: Optional[float] = 1.) -> None:
        """
        This function adds a set of g-nets to the g-net dictionary.

        Args:
            name (str): Name of the layer parameter predicted by the g-net.
            global_rank (int): Rank of the device where the g-net is stored.
            param (Tensor): The parameter tensor of the layer.
            row_col_encodings (torch.Tensor): The row and column encodings of 
                the parameter matrix.
            gnets (Sequence[HyperNetwork]): The g-net model.
            tile_shape (Tuple[int, int]): The shape of the tiles used to 
                predict the weights of the layer.
            output_scale (float): The scaling factor for the g-net output.
            grad_scale (float): The scaling factor for the gradients.
        """
        # TODO global_rank will have to be converted into local rank first!
        local_rank = self.local_rank_dict[global_rank]
        gnets = [gnet.to(local_rank) for gnet in gnets]
        row_col_encodings = row_col_encodings.to(local_rank)
        num_layers = len(gnets[0].sizes) # number of layers in a g-net
        ########################################################################
        # NOTE: Do not touch! Normalization has been carefully computed...     #
        ########################################################################
        _lr = self.lr / (num_layers - 1) ** 0.5 / output_scale.item() ** 0.5 
        ########################################################################

        optimizer = lambda params: optim.Adam(params,  lr=_lr, fused=True) # optim.SGD(params, lr=_lr, fused=True) # 
        optimizers = [optimizer(gnet.parameters()) for gnet in gnets]
        
        scheduler = self.scheduler
        schedulers = [scheduler(optimizers[i], _lr) for i in range(len(optimizers))]
        
        self.gnetdict[name] = GNetLayer(name=name, rank=global_rank,
                                        tile_shape=tile_shape, gnets=gnets,
                                        optimizers=optimizers,
                                        schedulers=schedulers,
                                        gnet_input=row_col_encodings,
                                        weights=param.data, grad_scale=grad_scale)
        
        total_params = sum(p.numel() for gnet in gnets for p in gnet.parameters())
        num_tiles = sum(gnet.num_tiles for gnet in gnets)
        print(f"Creating g-net for layer: {name}\n"
              f"Layer size: {param.shape}\n"
              f"Device ID: {global_rank}\n"
              f"Number of g-nets: {len(gnets)}\n"
              f"Total params per gnet: {total_params // num_tiles}\n"
              f"Learning rate: {_lr}\n")
        
