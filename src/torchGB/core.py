import time
from typing import Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor

from .utils import find_layer, assemble_matrix
from .gnet import GenomicBottleNet, conv2d_gnet_layer, default_gnet_layer, qkv_gnet_layer
from .lamb import Lamb


@dataclass
class GNetLayer:
    """
    This class stores all the information about the G-Net for a specific layer.
    
    Args:
        `name` (str): Name of the layer parameter predicted by the G-Net.
        `rank` (int): Rank of the device where the G-Net is stored.
        `gnets` (GenomicBottleNet): Sequence of G-Net models. This is typically 
                                    a MLP.
        `optimizer` (optim.Optimizer): The optimizer used to train the G-Net.
        `gnet_input` (Tensor): The input to the G-Net. This is a constant tensor
                            that is used to predict the new weights of the layer.
                            It encodes the (i,j)-position of every weight in the
                            parameter matrix of the layer.
        `weights` (Tensor): The new weights predicted by the G-Net.
        `grad_scale` (float): The scaling factor for the gradients.   
    """
    name: str
    rank: int
    tile_shape: Optional[Tuple[int, int]] = None
    gnets: Optional[Sequence[GenomicBottleNet]] = None
    optimizers: Optional[Sequence[optim.Optimizer]] = None
    gnets_inputs: Optional[Sequence[Tensor]] = None
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
        `hidden_dim` (int): The size of the hidden layers in the G-Nets.
        `ignore_layers` (Optional[Sequence[str]]): A list of layer names and types
                                                that should not be predicted
                                                using a G-Net.
    """
    gnetdict: Dict[str, GNetLayer]
    
    def __init__(self, 
                model: nn.Module, 
                hidden_dim: int = 64, 
                lr: float = 2.5-4,
                max_gnet_batch: int = 36_864,
                ignore_layers: Optional[Sequence[str]] = []) -> None:
        super(GenomicBottleneck, self).__init__()             
        
        # Stores all the information about the gnets
        self.gnetdict = {}
        load_per_rank = np.zeros(dist.get_world_size())  
        i0 = 0
                                      
        for pname, param in model.named_parameters():            
            ignore_layer = any([layer_name in pname for layer_name in ignore_layers])
            if param.requires_grad and not ignore_layer:
                # This implements a rudimentary load balancer across devices
                # that removes the bias towards the first device
                device_id = np.where(load_per_rank == load_per_rank.min())[0][-1]
                # if device_id == 0:
                #     if i0 > 2:
                #         device_id = np.where(load_per_rank[1:] == load_per_rank[1:].min())[0][-1]
                #         device_id += 1
                #     else:
                #         i0 += 1
                load_per_rank[device_id] += param.data.numel()
                
                if device_id == dist.get_rank():
                    print("Creating G-Net for layer:", pname)
                    print("Layer size:", np.array(param.shape))
                    print("Device ID:", device_id)   

                    # Find layer with that parameter name in the model
                    layer = find_layer(model, pname)    
                    grad_scale = torch.tensor(1.) 
                                          
                    # Normalizes the output to follow the initial parameter
                    # distribution at initialization of the model                  
                    with torch.no_grad():
                        output_scale = torch.std(param.data).to(device_id)
                    
                    param_shape = np.flip(np.array(param.data.shape))
                    
                    if isinstance(layer, nn.Conv2d):
                        if "weight" in pname:
                            row_col_encoding, gnet = conv2d_gnet_layer(param_shape, 
                                                                        hidden_dim,
                                                                        output_scale,
                                                                        max_gnet_batch)    
                    elif "in_proj_weight" in pname:
                        row_col_encodings, gnets, tile_shape = qkv_gnet_layer(param_shape, 
                                                                                hidden_dim,
                                                                                output_scale,
                                                                                max_gnet_batch)  
                        gnets = [gnet.to(device_id) for gnet in gnets]
                        optimizers = [Lamb(gnet.parameters(), lr=lr) for gnet in gnets]
                                    
                    else:                        
                        if param.data.ndim == 2:
                            # Treat 2D weight as fully connected                            
                            row_col_encodings, gnets, tile_shape = default_gnet_layer(param_shape, 
                                                                                        hidden_dim,
                                                                                        output_scale,
                                                                                        max_gnet_batch)
                            gnets = [gnet.to(device_id) for gnet in gnets]
                            optimizers = [Lamb(gnet.parameters(), lr=lr) for gnet in gnets]
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
                    self.gnetdict[pname] = GNetLayer(name=pname,
                                                    rank=device_id,
                                                    tile_shape=tile_shape,
                                                    gnets=gnets,
                                                    optimizers=optimizers,
                                                    gnets_inputs=row_col_encodings.to(device_id),
                                                    weights=param.data,
                                                    grad_scale=grad_scale.to(device_id))
                else:
                    self.gnetdict[pname] = GNetLayer(name=pname, rank=device_id)
                                            
    def __repr__(self) -> str:
        output_str = f"G-Net parameters:\n"
        for name in self.gnetdict.keys():
            output_str += f"Parameter={name}\n" \
                        f"Parameter shape={self.gnetdict[name].weights.shape}\n" \
                        f"G-Net input shape={self.gnetdict[name].gnet_inputs.shape}\n\n"
        return output_str
                                                                         
    def get_num_params_gnet(self) -> int:
        """
        Because gnets are now stored decentralized across devices, we need to
        compute them separately and then sum them up with a all_reduce operation.
        
        Returns:
            int: Cumulative number of parameters of all G-Nets.
        """
        num_params = torch.tensor([0]).to(dist.get_rank()) 
        
        for name in self.gnetdict.keys():
            n = 0
            if self.gnetdict[name].gnets is not None:
                for gnet in self.gnetdict[name].gnets:
                    n += sum(param.numel() for _, param in gnet.named_parameters())
            num_params += torch.tensor([n]).to(dist.get_rank())
        
        dist.all_reduce(num_params, op=dist.ReduceOp.SUM)
        return num_params.item()
    
    def compression(self, model) -> float:
        """
        `WARNING`: This function is currently totally inaccurate!
        This function computes the compression ratio of the G-Net to the model.

        Returns:
            float: Compression factor of the G-Net with respect to the model.
        """
        num_model_params = sum(p.numel() for p in model.parameters())
        num_gnet_params = self.get_num_params_gnet()
        
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
                
                for gnet, opt, gnet_params, opt_state in zip(d.gnets, d.optimizers, checkpoint[model_name], checkpoint[optimizer_name]):
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
        
    def predict_weights(self, model: nn.Module) -> None:
        """
        This function generates the new weights using the gnets, reshapes them
        to the original shape and sets the models parameters to the corresponding
        new weights.

        Args:
            `model` (nn.Module): The neural network model in question.
        """
        param_list = {i:[] for i in range(dist.get_world_size())}
        for name, param in model.named_parameters():
            if name in self.gnetdict.keys():
                if self.gnetdict[name].rank == dist.get_rank():
                    new_weights = []
                    tile_shape = self.gnetdict[name].tile_shape
                    for gnet_input, gnet in zip(self.gnetdict[name].gnets_inputs, self.gnetdict[name].gnets):
                        new_weight_tile = gnet(gnet_input)
                        new_weight_tile = new_weight_tile.reshape(tile_shape)
                        new_weights.append(new_weight_tile)
                    new_weights = torch.stack(new_weights)
                    new_weights = assemble_matrix(new_weights, param.data.shape)
                    self.gnetdict[name].weights = new_weights
                    # Sets the models parameters to the corresponding new weights
                    param.data = nn.Parameter(new_weights)
                param_list[self.gnetdict[name].rank].append(param.data)
                
        for source_id in range(dist.get_world_size()):
            for j in range(len(param_list[source_id])):
                # Broadcast the weights of the gnets calculated on GPU with
                # rank `dist.get_rank()` to all other GPUs.
                dist.broadcast(param_list[source_id][j], src=source_id)
    
    def backward(self, model: nn.Module) -> None:
        """
        This function takes the models gradients after a forward and 
        backward pass through the model and propagates them through the Gnet to
        update the parameters.

        Args:
            `model` (nn.Module): The neural network model.
        """              
        for name, param in model.named_parameters():
            if name in self.gnetdict.keys():
                if self.gnetdict[name].rank == dist.get_rank():
                    grad_scale = self.gnetdict[name].grad_scale
                    norm_grad = torch.div(param.grad, grad_scale)
                    self.gnetdict[name].weights.backward(norm_grad)
                      
    def step(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                for optimizer in self.gnetdict[name].optimizers:
                    optimizer.step()

