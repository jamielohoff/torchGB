import time
from typing import Callable, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor

from .gnet import GenomicBottleNet

from .layers.attn_gnet import init_attn_gnet, build_attn_gnet_output
from .layers.conv_gnet import init_conv2d_gnet, build_conv2d_gnet_output
from .layers.linear_gnet import init_linear_gnet, build_linear_gnet_output


# Stores how the compression is intended to work for different layer types
gnet_types = {}

@dataclass
class GNetType:
    """
    Simple container data structure to store the init and build functions of a 
    g-net.

    Args:
        _type_: _description_
    """
    name: str
    init: Callable
    build: Callable
    

def register_gnet_type(mod_type: nn.Module, init: Callable, build: Callable) -> None:
    """
    This function registers a new g-net type to be used in the Genomic Bottleneck.

    Args:
        type (nn.Module): _description_
        init (Callable): _description_
        build (Callable): _description_
    """
    global gnet_types
    gnet_types[mod_type] = GNetType(str(mod_type), init, build)
    
    
# TODO registering does not work as intended yet
register_gnet_type(nn.TransformerEncoder, init_attn_gnet, build_attn_gnet_output)
register_gnet_type(nn.Linear, init_linear_gnet, build_linear_gnet_output)
register_gnet_type(nn.Conv2d, init_conv2d_gnet, build_conv2d_gnet_output)


@dataclass
class GNetLayer:
    """
    This class stores all the information about the g-net for a specific layer.
    
    Args:
        `name` (str): Name of the layer parameter predicted by the g-net.
        `rank` (int): Rank of the device where the g-net is stored.
        `tile_shape` (Optional[Sequence[int, int]]): The shape of the tiles used 
            to predict the weights of the layer.
        `gnets` (Sequence[GenomicBottleNet]): Sequence of g-net models. 
            This is typically a list of MLPs.
        `optimizers` (Sequence[optim.Optimizer]): The optimizers used to train 
            all the the g-nets.
        `gnet_input` (Tensor): The input to the g-net. This is a 
            constant tensor that is used to predict the new weights of the 
            layer. They encode the (i,j)-position of every weight in the 
            parameter matrix of the layer.
        `weights` (Tensor): The weights predicted by the g-net.
        `grad_scale` (float): The scaling factor for the gradients.   
    """
    name: str
    rank: int
    tile_shape: Optional[Sequence[int]] = None
    gnets: Optional[Sequence[GenomicBottleNet]] = None
    optimizers: Optional[Sequence[optim.Optimizer]] = None
    schedulers: Optional[optim.lr_scheduler.LRScheduler] = None
    gnet_input: Optional[Sequence[Tensor]] = None
    weights: Optional[Tensor] = None
    grad_scale: Optional[float] = None


class GenomicBottleneck(nn.Module):
    """
    The `GenomicBottleneck` class implements a hypernetwork that predicts all
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
    When launched with the `torchrun -nproc_per_node='num_gpus'` command, 
    every g-net is stored on a different device to parallelize the computation. 
    Furthermore, every g-net has its own optimizer.
    Gradients are backpropagated by first backpropagating the gradients through
    the `model` and then using them as seeds for further backpropagation through
    the g-nets.

    Args:
        `model` (nn.Module): The neural network model.
        `num_batches` (int): The number of batches in the training loop.
        `hidden_dim` (int): The size of the hidden layers in the g-nets.
        `lr` (float): The learning rate of the g-nets.
        `gnet_batchsize` (int): The number of parameters per tile.
        `ignore_layers` (Optional[Sequence[str]]): A list of layer names and 
            types that should not be predicted using a g-net.
    """
    lr: float
    num_batches: int
    model: nn.Module
    gnetdict: Dict[str, GNetLayer]
    
    def __init__(self, model: nn.Module, num_batches: int = 0, 
                 hidden_dim: int = 32, lr: float = 0.001, 
                 gnet_batchsize: int = 10_000, 
                 ignore_layers: Sequence[str] = []) -> None:
        super(GenomicBottleneck, self).__init__()             
        self.model = model
        self.lr = lr
        self.num_batches = num_batches
        
        # Stores all the information about the gnets
        self.gnetdict = {}
        initialized = set()
        load_per_rank = np.zeros(dist.get_world_size()) 
           
        # Iterate over all the modules in the model
        for name, mod in self.model.module.named_modules(): 
            for pname, param in mod.named_parameters():    
                _name = name + "." + pname 
                ignore_param = any([lname in _name for lname in ignore_layers])
                if param.requires_grad and not ignore_param and not "bias" in pname and not _name in initialized:
                    # This implements a rudimentary load balancer across devices
                    # that removes the bias towards the first device
                    device_id = np.where(load_per_rank == load_per_rank.min())[0][-1]
                    load_per_rank[device_id] += param.data.numel()
                    
                    # NOTE: Edit these lines in order to add a new layers for
                    # compression or edit the way the current compression behavior
                    gnet_type = gnet_types.get(type(mod))
                    if gnet_type:
                        # First index of compression behavior describes 
                        # initialization of the g-net for this specific
                        # layer type
                        if device_id == dist.get_rank():
                            out_fn = gnet_type.init
                            out = out_fn(pname, param, hidden_dim, 
                                            gnet_batchsize)
                            if out_fn:
                                self._add_gnets(_name, device_id, param, *out)
                            
                        else:
                            self.gnetdict[_name] = GNetLayer(name=_name, rank=device_id)
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
        num_params = torch.tensor(0).to(dist.get_rank()) 
        
        for name in self.gnetdict.keys():
            n = 0
            if self.gnetdict[name].gnets is not None:
                for gnet in self.gnetdict[name].gnets:
                    n += sum(param.numel() for _, param in gnet.named_parameters())
            num_params += torch.tensor(n).to(dist.get_rank())
        
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
            `fname` (str): File to which we wish to write the weights of the g-nets.
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
            `fname` (str): File from which to load the g-nets.
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

    def train(self) -> None:
        """
        Trains the neural networks in the `gnetdict` attribute that are assigned 
        to the current process rank. This method iterates over the keys in the 
        `gnetdict` attribute and checks if the `rank` of each `gnetdict` entry
        matches the current process rank. If it does, it sets the corresponding 
        neural networks to training mode.
        """
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for gnet in self.gnetdict[name].gnets:
                    gnet.train()
    
    def zero_grad(self) -> None:
        """
        Zeros out all gradients in the optimizers, similarly to `loss.zero_grad()`.
        This function iterates over all the networks in the `gnetdict` dictionary.
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
        if dist.get_rank() == 0:
            print(self.gnetdict.get("transformer_encoder.layers.4.self_attn.in_proj_weight").gnets[0].layers[0].weight.data)
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
                            gnet_input = gnetstack.gnet_input
                            new_weight_tile = gnet(gnet_input)
                            new_weight_tile = new_weight_tile.reshape(tile_shape)
                            new_weights.append(new_weight_tile)
                    
                        # Assemble the new weight tiles into the full weight matrix
                        new_weights = torch.stack(new_weights, dim=0)
                        
                        # Second index of compression behavior describes 
                        # assembly of the weight matrix predicted by
                        # the g-nets
                        gnet_type = gnet_types.get(type(mod))
                        if gnet_type:
                            weights_fn = gnet_type.build
                            new_weights = weights_fn(_name, param, new_weights, 
                                                     tile_shape)
                            
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
                # rank `dist.get_rank()` to all other GPUs.
                dist.broadcast(param_list[source_id][j], src=source_id)
    
    def backward(self) -> None:
        """
        This function takes the models gradients after a forward and 
        backward pass through the model and propagates them through the g-net to
        update the parameters.
        """           
        # We need to find a way so that we only backpropagate through every layer
        # once TODO
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
                            print("backpropagating", _name)
                            gnetstack.weights.backward(norm_grad)
                            backpropagated.add(_name)
                      
    def step(self) -> None:
        """
        Performs a single optimization step for each optimizer in the network 
        dictionary.

        This function iterates over the keys in the `gnetdict` attribute, 
        which is a dictionary containing network objects. For each network 
        object that matches the current process rank, it performs an 
        optimization step using the optimizers associated with that network.

        Note:
            - The function currently only steps the optimizers and not the 
                schedulers, as the scheduler step code is commented out.

        Attributes:
            gnetdict (dict): A dictionary where keys are network names and values 
                are network objects that contain optimizers and schedulers.
        """
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                # for optimizer, scheduler in zip(self.gnetdict[name].optimizers, 
                #                                 self.gnetdict[name].schedulers):
                for optimizer in self.gnetdict[name].optimizers:
                    optimizer.step()
                    # scheduler.step()
                    
    def _add_gnets(self, name: str, device_id: int, param: Tensor,
                   row_col_encodings: Tensor, gnets: GenomicBottleNet, 
                   tile_shape: Tuple[int, int], output_scale: float,
                   grad_scale: Optional[float] = 1.) -> None:
        """
        This function adds a set of g-nets to the g-net dictionary.

        Args:
            `name` (str): Name of the layer parameter predicted by the g-net.
            `device_id` (int): Rank of the device where the g-net is stored.
            `param` (Tensor): The parameter tensor of the layer.
            `row_col_encodings` (torch.Tensor): The row and column encodings of 
                the parameter matrix.
            `gnets` (Sequence[GenomicBottleNet]): The g-net model.
            `tile_shape` (Tuple[int, int]): The shape of the tiles used to 
                predict the weights of the layer.
            `output_scale` (float): The scaling factor for the g-net output.
            `grad_scale` (float): The scaling factor for the gradients.
        """
        gnets = [gnet.to(device_id) for gnet in gnets]
        row_col_encodings = row_col_encodings.to(device_id)
        
        num_layers = len(gnets[0].sizes) # number of layers in a gnet
        # NOTE: Do not touch! Normalization has been carefully computed...
        _lr = self.lr / (num_layers - 1) ** 0.5 / output_scale.item() ** 0.5

        optimizer = lambda params: optim.SGD(params, lr=_lr)
        optimizers = [optimizer(gnet.parameters()) for gnet in gnets]
        
        # scheduler = lambda optim: OneCycleLR(optim,
        #                                     max_lr=_lr, 
        #                                     pct_start=0.2,
        #                                     div_factor=250,
        #                                     final_div_factor=1000,
        #                                     total_steps=self.num_batches)
        
        # schedulers = [scheduler(optim) for optim in optimizers]
        
        grad_scale = 1. # placeholder value; never used
        self.gnetdict[name] = GNetLayer(name=name, rank=device_id,
                                        tile_shape=tile_shape, gnets=gnets,
                                        optimizers=optimizers,
                                        # schedulers=schedulers,
                                        gnet_input=row_col_encodings,
                                        weights=param.data,
                                        grad_scale=grad_scale)
        
        print(f"Creating g-net for layer: {name}\n"
              f"Layer size: {param.shape}\n"
              f"Device ID: {device_id}\n"
              f"Number of g-nets: {len(gnets)}\n"
              f"Learning rate: {_lr}\n")
        
