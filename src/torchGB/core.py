import time
from typing import Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor

from torch.optim.lr_scheduler import OneCycleLR

from .utils import assemble_matrix, cut_matrix, assemble_4d_kernel
from .gnet import (GenomicBottleNet, 
                    square_conv2d_gnet_layer,
                    square_default_gnet_layer, 
                    square_qkv_gnet_layer,
                    onedim_gnet_layer,
                    ceil)
# from lamb import Lamb


@dataclass
class GNetLayer:
    """
    This class stores all the information about the G-Net for a specific layer.
    
    Args:
        `name` (str): Name of the layer parameter predicted by the G-Net.
        `rank` (int): Rank of the device where the G-Net is stored.
        `tile_shape` (Optional[Sequence[int, int]]): The shape of the tiles used 
            to predict the weights of the layer.
        `gnets` (Sequence[GenomicBottleNet]): Sequence of G-Net models. 
            This is typically a list of MLPs.
        `optimizers` (Sequence[optim.Optimizer]): The optimizers used to train 
            all the the G-Nets.
        `gnet_input` (Tensor): The input to the G-Net. This is a 
            constant tensor that is used to predict the new weights of the 
            layer. They encode the (i,j)-position of every weight in the 
            parameter matrix of the layer.
        `weights` (Tensor): The weights predicted by the G-Net.
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
    a G-Net is created that predicts the new weights of the layer.
    When launched with the `-m torch.distributed.run` command, every G-Net is 
    stored on a different device to parallelize the computation. Furthermore,
    every G-Net has its own optimizer.
    Gradients are backpropagated by first backpropagating the gradients through
    the `model` and then using them as seeds for further backpropagation through
    the G-Nets.

    Args:
        `model` (nn.Module): The neural network model.
        `num_batches` (int): The number of batches in the training loop.
        `hidden_dim` (int): The size of the hidden layers in the G-Nets.
        `lr` (float): The learning rate of the G-Nets.
        `max_gnet_batchsize` (int): The maximum number of parameters per tile.
        `ignore_layers` (Optional[Sequence[str]]): A list of layer names and 
            types that should not be predicted using a G-Net.
    """
    lr: float
    num_batches: int
    model: nn.Module
    gnetdict: Dict[str, GNetLayer]
    
    def __init__(self, 
                model: nn.Module, 
                num_batches: int = 0,
                hidden_dim: int = 32, 
                lr: float = 1e-3,
                max_gnet_batchsize: int = 36_864,
                ignore_layers: Sequence[str] = []) -> None:
        super(GenomicBottleneck, self).__init__()             
        self.model = model
        self.lr = lr
        self.num_batches = num_batches
        
        # Stores all the information about the gnets
        self.gnetdict = {}
        load_per_rank = np.zeros(dist.get_world_size())       
        for name, mod in model.module.named_modules():
            if (not isinstance(mod, nn.Linear)) and (not isinstance(mod, nn.Conv2d)):
                continue
            for pname, param in mod.named_parameters():    
                _name = name + "." + pname 
                ignore_param = any([lname in _name for lname in ignore_layers])
                if param.requires_grad and not ignore_param:
                    # This implements a rudimentary load balancer across devices
                    # that removes the bias towards the first device
                    device_id = np.where(load_per_rank == load_per_rank.min())[0][-1]
                    load_per_rank[device_id] += param.data.numel()
                    
                    if device_id == dist.get_rank():
                        if isinstance(mod, nn.Conv2d):
                            if "weight" in pname:
                                out = square_conv2d_gnet_layer(param, 
                                                                hidden_dim,
                                                                max_gnet_batchsize)   
                                # TODO reintegrate this
                                # Add layer to the dict                                                          
                                # pname_cut = pname.split("weight")[0] # that's a sloppy way to do that
                                # pname_cut = pname_cut.split("bias")[0]
                                # for name_tmp, layer_tmp in model.named_modules():
                                #     if name_tmp == pname_cut:
                                #         _out_size = get_tensor_dimensions(model, layer_tmp, input_shape)
                                #         _out_size = torch.tensor(_out_size)
                                        
                                # if isinstance(layer, nn.Conv2d):
                                #     grad_scale = _out_size[-1][-1] 
                                self._add_gnets(_name, device_id, param, *out)
                                
                        elif isinstance(mod, nn.TransformerEncoder):
                            if "in_proj_weight" in pname:
                                out = square_qkv_gnet_layer(param, 
                                                            hidden_dim,
                                                            max_gnet_batchsize) 

                                self._add_gnets(_name, device_id, param, *out)
                            elif "weight" in pname and len(param.shape) == 2:
                                out = square_default_gnet_layer(param, 
                                                                hidden_dim,
                                                                max_gnet_batchsize) 
                                self._add_gnets(_name, device_id, param, *out)
                            elif "bias" in pname:
                                out = onedim_gnet_layer(param, 
                                                        hidden_dim,
                                                        max_gnet_batchsize)
                                self._add_gnets(_name, device_id, param, *out)
                                        
                        elif isinstance(mod, nn.Linear):     
                            if "weight" in pname:                   
                                # Treat everything else as fully connected                            
                                out = square_default_gnet_layer(param, 
                                                                hidden_dim,
                                                                max_gnet_batchsize)
                                self._add_gnets(_name, device_id, param, *out)
                            else:
                                out = onedim_gnet_layer(param, 
                                                        hidden_dim,
                                                        max_gnet_batchsize)
                                self._add_gnets(_name, device_id, param, *out)
                    else:
                        self.gnetdict[_name] = GNetLayer(name=_name, rank=device_id)
                    
    def __repr__(self) -> str:
        output_str = f"G-Net parameters:\n"
        for name in self.gnetdict.keys():
            gnets = self.gnetdict[name]
            output_str += f"Parameter={name}\n" \
                        f"Parameter shape={gnets.weights.shape}\n" \
                        f"G-Net input shape={gnets.gnet_input.shape}\n\n"
        return output_str
                                                                         
    def get_num_params_gnet(self) -> int:
        """
        Because gnets are now stored decentralized across devices, we need to
        compute them separately and then sum them up with a all_reduce operation.
        
        Returns:
            int: Cumulative number of parameters of all G-Nets.
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
            int: Cumulative number of parameters of which have no G-Nets attached.
        """
        return sum([param.numel() for pname, param in self.model.named_parameters()
                    if not pname in self.gnetdict.keys()])
    
    def compression(self) -> float:
        """
        This function computes the compression ratio of the G-Net to the model.

        Returns:
            float: Compression factor of the G-Net with respect to the model.
        """
        num_model_params = sum(p.numel() for p in self.model.parameters())
        num_gnet_params = self.get_num_params_gnet() 
        num_gnet_params += self.get_num_params_no_gnet()
        
        compression = num_model_params / num_gnet_params
        return compression
                    
    def save(self, fname: str) -> None:
        """
        Save the G-Nets from a checkpoint file.

        Args:
            `fname` (str): File to which we wish to write the weights of the G-Nets.
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
        Load the G-Nets from a checkpoint file.

        Args:
            `fname` (str): File from which to load the G-Nets.
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
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for gnet in self.gnetdict[name].gnets:
                    gnet.train()
    
    def zero_grad(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for gnet in self.gnetdict[name].gnets:
                    gnet.zero_grad()
        
    def predict_weights(self) -> None:
        """
        This function generates the new weights using the gnets, reshapes them
        to the original shape and sets the models parameters to the corresponding
        new weights.
        """
        param_list = {i:[] for i in range(dist.get_world_size())}
        for name, mod in self.model.module.named_modules(): 
            if (not isinstance(mod, nn.Linear)) and (not isinstance(mod, nn.Conv2d)):
                continue
            for pname, param in mod.named_parameters():    
                _name = name + "." + pname 
                if _name in self.gnetdict.keys():
                    if self.gnetdict[_name].rank == dist.get_rank():
                        new_weights = []
                        tile_shape = self.gnetdict[_name].tile_shape
                        for gnet in self.gnetdict[_name].gnets:
                            gnet_input = self.gnetdict[_name].gnet_input
                            new_weight_tile = gnet(gnet_input)
                            new_weight_tile = new_weight_tile.reshape(tile_shape)
                            # new_weight_tile -= new_weight_tile.mean()
                            new_weights.append(new_weight_tile)
                            
                        # TODO make this prettier
                        # Assemble the new weight tiles into the full weight matrix
                        new_weights = torch.stack(new_weights, dim=0)
                        
                        if "in_proj_weight_x" in _name:
                            num_row_tiles = 3*ceil(param.shape[0]//3/tile_shape[0])
                            num_col_tiles = ceil(param.shape[1]/tile_shape[1])
                        elif "bias" in _name:
                            num_col_tiles = num_row_tiles = 1
                        else:
                            num_row_tiles = ceil(param.shape[0]/tile_shape[0])
                            num_col_tiles = ceil(param.shape[1]/tile_shape[1])
                            
                        if isinstance(mod, nn.Conv2d):
                            shape = (num_row_tiles*tile_shape[0], 
                                        num_col_tiles*tile_shape[1], 
                                        param.shape[2], 
                                        param.shape[3])
                            new_weights = assemble_4d_kernel(new_weights, shape)
                            new_weights = cut_matrix(new_weights, param.shape)
                        elif "bias" in _name:
                            new_weights = new_weights.squeeze()
                        else:
                            shape = (num_row_tiles*tile_shape[0], 
                                    num_col_tiles*tile_shape[1])
                            new_weights = assemble_matrix(new_weights, shape)
                            new_weights = cut_matrix(new_weights, param.shape)
                            
                        # When cutting the matrix, it's not contiguous anymore
                        new_weights = new_weights.contiguous() 
                        self.gnetdict[_name].weights = new_weights
                        
                        # Sets the parameters to the corresponding new weights
                        param.data = nn.Parameter(new_weights)
                    param_list[self.gnetdict[_name].rank].append(param.data)
                
        for source_id in range(dist.get_world_size()):
            for j in range(len(param_list[source_id])):
                # Broadcast the weights of the gnets calculated on GPU with
                # rank `dist.get_rank()` to all other GPUs.
                dist.broadcast(param_list[source_id][j], src=source_id)
    
    def backward(self) -> None:
        """
        This function takes the models gradients after a forward and 
        backward pass through the model and propagates them through the G-Net to
        update the parameters.
        """              
        for name, mod in self.model.module.named_modules(): 
            if (not isinstance(mod, nn.Linear)) and (not isinstance(mod, nn.Conv2d)):
                continue
            for pname, param in mod.named_parameters():  
                _name = name + "." + pname 
                if _name in self.gnetdict.keys():
                    if self.gnetdict[_name].rank == dist.get_rank():
                        grad_scale = self.gnetdict[_name].grad_scale
                        norm_grad = param.grad / grad_scale
                        self.gnetdict[_name].weights.backward(norm_grad)
                      
    def step(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                # for optimizer, scheduler in zip(self.gnetdict[name].optimizers, 
                #                                 self.gnetdict[name].schedulers):
                for optimizer in self.gnetdict[name].optimizers:
                    optimizer.step()
                    # scheduler.step()
                    
    def _add_gnets(self, 
                    name: str, 
                    device_id: int,
                    param: Tensor,
                    row_col_encodings: Tensor,
                    gnets: GenomicBottleNet, 
                    tile_shape: Tuple[int, int],
                    output_scale: float,
                    grad_scale: Optional[float] = 1.) -> None:
        """
        This function adds a set of G-Nets to the G-Net dictionary.

        Args:
            `name` (str): Name of the layer parameter predicted by the G-Net.
            `device_id` (int): Rank of the device where the G-Net is stored.
            `param` (Tensor): The parameter tensor of the layer.
            `row_col_encodings` (torch.Tensor): The row and column encodings of 
                the parameter matrix.
            `gnets` (Sequence[GenomicBottleNet]): The G-Net model.
            `tile_shape` (Tuple[int, int]): The shape of the tiles used to 
                predict the weights of the layer.
            `output_scale` (float): The scaling factor for the output of the G-Net.
            `grad_scale` (float): The scaling factor for the gradients.
        """
        gnets = [gnet.to(device_id) for gnet in gnets]
        row_col_encodings = row_col_encodings.to(device_id)
        
        num_layers = len(gnets[0].sizes)
        _lr = self.lr / np.sqrt(output_scale.item()*num_layers)
        optimizer = lambda params: torch.optim.SGD(params, lr=_lr)
        optimizers = [optimizer(gnet.parameters()) for gnet in gnets]
        
        # scheduler = lambda optim: OneCycleLR(optim,
        #                                     max_lr=_lr, 
        #                                     pct_start=0.2,
        #                                     div_factor=250,
        #                                     final_div_factor=1000,
        #                                     total_steps=self.num_batches)
        
        # schedulers = [scheduler(optim) for optim in optimizers]
        
        grad_scale = torch.tensor(grad_scale).to(device_id)        
        self.gnetdict[name] = GNetLayer(name=name,
                                        rank=device_id,
                                        tile_shape=tile_shape,
                                        gnets=gnets,
                                        optimizers=optimizers,
                                        # schedulers=schedulers,
                                        gnet_input=row_col_encodings,
                                        weights=param.data,
                                        grad_scale=grad_scale)
        
        print(f"Creating G-Net for layer: {name}\n"
                f"Layer size: {param.shape}\n"
                f"Device ID: {device_id}\n"
                f"Number of g-nets: {len(gnets)}\n"
                f"Learning rate: {_lr}\n")

