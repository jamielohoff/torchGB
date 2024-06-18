from typing import Dict, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from .utils import find_layer
from .gnet import GenomicBottleNet, conv2d_gnet_layer, default_gnet_layer


class GNetLayer:
    """
    This class stores all the information about the G-Net for a specific layer.
    
    Args:
        `name` (str): Name of the layer parameter predicted by the G-Net.
        `rank` (int): Rank of the device where the G-Net is stored.
        `gnet` (GenomicBottleNet): The G-Net model. This is typically a MLP.
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
    gnet: GenomicBottleNet
    optimizer: optim.Optimizer
    gnet_input: Tensor
    weights: Tensor
    new_weights: Tensor
    grad_scale: float
    
    def __init__(self, 
                name: str, 
                rank: int,
                gnet: Optional[GenomicBottleNet] = None, 
                optimizer: Optional[optim.Optimizer] = None, 
                gnet_input: Optional[Tensor] = None, 
                weights: Optional[Tensor] = None, 
                grad_scale: Optional[float] = None) -> None:
        self.rank = rank
        self.name = name
        self.gnet = gnet
        self.optimizer = optimizer
        self.gnet_input = gnet_input
        self.weights = weights
        self.grad_scale = grad_scale


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
                ignore_layers: Optional[Sequence[str]] = []) -> None:
        super(GenomicBottleneck, self).__init__()             
        
        # Stores all the information about the gnets
        self.gnetdict = {}
        load_per_rank = [0 for _ in range(dist.get_world_size())]   
        
        # TODO this needs some disentanglement                                   
        for pname, param in model.named_parameters():            
            ignore_layer = any([layer_name in pname for layer_name in ignore_layers])
            if param.requires_grad and not ignore_layer:
                # This implements a rudimentary load balancer across devices
                device_id = min(enumerate(load_per_rank), key=lambda x: x[1])[0]
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
                                                                        output_scale)
                    else:                        
                        if param.data.ndim == 2:
                            # Treat 2D weight as fully connected
                            row_col_encoding, gnet = default_gnet_layer(param_shape, 
                                                                        hidden_dim,
                                                                        output_scale)
                                
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
                                                    gnet=gnet.to(device_id),
                                                    optimizer=optim.Adam(gnet.parameters()),
                                                    gnet_input=row_col_encoding.to(device_id),
                                                    weights=param.data,
                                                    grad_scale=grad_scale.to(device_id))
                else:
                    self.gnetdict[pname] = GNetLayer(name=pname, rank=device_id)
                                            
    def __repr__(self) -> str:  
        output_str = f"G-Net parameters:\n"
        for name in self.gnetdict.keys():
            output_str += f"Parameter={name}\n" \
                        f"Parameter shape={self.gnetdict[name].new_weights.shape}\n" \
                        f"G-Net input shape={self.gnetdict[name].gnet_input.shape}\n\n"
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
            gnet = self.gnetdict[name].gnet
            if gnet is not None:
                n = sum(param.numel() for _, param in gnet.named_parameters())
            else:
                n = 0
            num_params += torch.tensor([n]).to(dist.get_rank())
        
        dist.all_reduce(num_params, op=dist.ReduceOp.SUM)
        return num_params.item()
    
    def compression(self, model) -> float:
        """
        `WARNING`: This function is currently totally inaccurate!
        This function computes the compression ratio of the G-Net to the model.

        Returns:
            float: Compression factor of the G-Net to the model.
        """
        num_model_params = sum(p.numel() for p in model.parameters())
        num_gnet_params = self.get_num_params_gnet()
        
        compression = num_model_params/num_gnet_params
        return compression
                        
    def save(self, fname: str) -> None:
        """
        Save the G-Nets from a checkpoint file.

        Args:
            `fname` (str): File to which we wish to write the weights of the G-Nets.
        """
        print("Saving G-Nets ...\n") 
        param_dict = {}
            
        for name in self.gnetdict.keys():     
            if self.gnetdict[name].rank == dist.get_rank():     
                entry_name = name + "_state_dict"
                model_name = "model_" + entry_name
                optimizer_name = "optimizer_" + entry_name
                param_dict[model_name] = self.gnetdict[name].gnet.state_dict()
                param_dict[optimizer_name] = self.gnetdict[name].optimizer.state_dict()
                        
        torch.save(param_dict, fname)
                      
    def load(self, fname: str) -> None:
        """
        Load the G-Nets from a checkpoint file.

        Args:
            `fname` (str): File from which to load the G-Nets.
        """
        print("Loading G-Nets ...\n")
        checkpoint = torch.load(fname)

        for name in self.gnetdict.keys():          
            if self.gnetdict[name].rank == dist.get_rank():        
                entry_name = name + "_state_dict"
                model_name = "model_" + entry_name
                optimizer_name = "optimizer_" + entry_name                
                self.gnetdict[name].gnet.load_state_dict(checkpoint[model_name])
                self.gnetdict[name].optimizer.load_state_dict(checkpoint[optimizer_name])
         
    def train(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                self.gnetdict[name].gnet.train()
    
    def zero_grad(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                self.gnetdict[name].optimizer.zero_grad()
        
    def predict_weights(self, model: nn.Module) -> None:
        """
        This function generates the new weights using the gnets, reshapes them
        to the original shape and sets the models parameters to the corresponding
        new weights.

        Args:
            `model` (_type_): The neural network model in question.
        """
        param_list = {i:[] for i in range(dist.get_world_size())}
        for name, param in model.named_parameters():
            if name in self.gnetdict.keys():
                if self.gnetdict[name].rank == dist.get_rank():
                    gnet_inputs = self.gnetdict[name].gnet_input
                    new_weights = self.gnetdict[name].gnet(gnet_inputs)
                    new_weights = new_weights.view(param.data.shape)
                    self.gnetdict[name].weights = new_weights
                    # Sets the models parameters to the corresponding new weights
                    param.data = nn.Parameter(new_weights)
                param_list[self.gnetdict[name].rank].append(param.data)
                
        # TODO Get rid of the nested for-loop and replace by sth. faster
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
            `model` (_type_): The neural network model.
        """
        for name, param in model.named_parameters():
            if name in self.gnetdict.keys():
                if self.gnetdict[name].rank == dist.get_rank():
                    device = self.gnetdict[name].grad_scale.device
                    norm_grad = torch.div(param.grad.to(device), self.gnetdict[name].grad_scale)
                    self.gnetdict[name].weights.backward(norm_grad)
                      
    def step(self) -> None:
        for name in self.gnetdict.keys():
            if self.gnetdict[name].rank == dist.get_rank():
                self.gnetdict[name].optimizer.step()

