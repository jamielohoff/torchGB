from typing import Dict, Optional, Sequence, Tuple

import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from src.torchGB.utils import find_layer, get_tensor_dimensions, generate_GDN_layer


NIL = 0 # No encoding
HOT = 1 # One-hot vector
BIN = 2 # Binary code
GRY = 3 # Gray code
LIN = 4 # Linear? code
RND = 5 # Random code


class GenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length Gnet that uses a for-loop fir 
    initialization. This is a more flexible version of the GDNx classes.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    output_scale: float
    layers: nn.ModuleList
    
    def __init__(self, 
                sizes: Sequence[int], 
                output_scale: float) -> None:
        assert len(sizes) > 1, "List must have at least 3 entries!"
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) 
                                    for i in range(length)])
        self.apply(self.init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = F.silu(layer(x))
        output_scale = self.output_scale if self.output_scale > 1e-8 else torch.tensor(1.).to(self.output_scale.device)
        return output_scale * x
    
    def init_weights(self, module: nn.Module) -> None:            
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=module.weight.data.shape[0])
            nn.init.normal_(module.bias, std=module.bias.data.shape[0])

class GNet:
    """
    Named tuple to store the GNet information.

    Args:
        _type_: _description_

    Returns:
        _type_: _description_
    """
    name: str
    gnet: GenomicBottleNet
    optimizer: optim.Adam
    gnet_input: Tensor
    weights: Tensor
    new_weights: Tensor
    output_scaling: float
    grad_scale: Tensor
    
    def __init__(self, 
                name: str, 
                gnet: GenomicBottleNet, 
                optimizer: optim.Optimizer, 
                gnet_input: Tensor, 
                weights: Tensor, 
                new_weights: Tensor, 
                grad_scale: float) -> None:
        self.name = name
        self.gnet = gnet
        self.optimizer = optimizer
        self.gnet_input = gnet_input
        self.weights = weights
        self.new_weights = new_weights
        self.grad_scale = grad_scale


class GNetList(nn.Module):
    """
    TODO docstring

    Args:
        nn (_type_): _description_
    """
    gnetdict: Dict[str, GNet]
    
    def __init__(self, 
                model: nn.Module, 
                num_hidden_layers: int = 50, 
                input_shape: Tuple[int, int] = (256, 256),
                ignore_layers: Optional[Sequence[str]] = [],
                gpu_list: Optional[Sequence[int]] = []):
        super(GNetList, self).__init__()             
        
        # Stores all the information about the gnets
        self.gnetdict = {}
                                                           
        for i, (pname, param) in enumerate(model.named_parameters()):            
            ignore_layer = any([layer_name in pname for layer_name in ignore_layers])
            if param.requires_grad and not ignore_layer:
                device_id = i % len(gpu_list)
                
                # find layer with that parameter name in the model
                layer = find_layer(model, pname)     
                                
                print("Creating G-net for layer:", pname)
                print("Layer size:", np.array(param.shape))
                print("Device ID:", device_id)
                    
                with torch.no_grad():
                    # Normalizes the output to follow the initial parameter
                    # distribution at initialization of the model
                    output_scale = torch.std(param.data) 
                                
                param_shape = np.flip(np.array(param.data.shape))
                encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
                encoding_bits[:2] = param_shape[:2]
                encoding_bits[np.where(encoding_bits == 0)] = 1
                
                if isinstance(layer, nn.Conv2d):
                    print("Conv2d Layer")
                    if "weight" in pname:
                        encoding_type = (HOT, HOT, BIN, BIN)                       
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers, num_hidden_layers, num_hidden_layers//2, 1)
                                            
                    if "bias" in pname:                                                        
                        encoding_type = (BIN,)
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers, 1)
                
                elif isinstance(layer, nn.Linear):
                    print("Fully connected Layer")
                    if "weight" in pname:
                        param_shape = np.flip(np.array(param.data.shape))
                        encoding_type = (BIN, BIN)
                        encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
                        encoding_bits[np.where(encoding_bits == 0)] = 1                        
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers, num_hidden_layers, num_hidden_layers//2, 1)
                                            
                    if "bias" in pname:
                        param_shape = np.flip(np.array(param.data.shape))
                        encoding_type = (BIN,)
                        encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
                        encoding_bits[np.where(encoding_bits == 0)] = 1
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers//2, 1)
                
                else:
                    layer_shape = np.array(param.data.shape)
                    layer_shape_size = layer_shape.size
                    param_shape = np.flip(np.array(param.data.shape))
                        
                    if layer_shape_size == 2:
                        # treat 2D weight as fully connected
                        param_shape = np.flip(np.array(param.data.shape))
                        encoding_type = (BIN, BIN)
                        encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
                        encoding_bits[np.where(encoding_bits == 0)] = 1
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers, num_hidden_layers//2, 1)
                            
                    if layer_shape_size == 1:       
                        # treat 1D shape as bias 
                        param_shape = np.flip(np.array(param.data.shape))
                        encoding_type = (BIN,)
                        encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
                        encoding_bits[np.where(encoding_bits == 0)] = 1
                        weight, row_col_encoding = generate_GDN_layer(encoding_type, param_shape, encoding_bits)
                        num_inputs = row_col_encoding.shape[1]
                        gnet_sizes = (num_inputs, num_hidden_layers//2, 1)

                # Add layer to the dict                                                          
                pname_cut = pname.split("weight")[0] # that's a sloppy way to do that
                pname_cut = pname_cut.split("bias")[0]
                for name_tmp, layer_tmp in model.named_modules():
                    if name_tmp == pname_cut:
                        _out_size = get_tensor_dimensions(model, layer_tmp, input_shape)
                        _out_size = torch.tensor(_out_size).to(device_id)
                        
                if isinstance(layer, nn.Conv2d):
                    grad_scale = _out_size[-1][-1]
                else:
                    grad_scale = torch.tensor(1.).to(device_id)
                    
                gnet = GenomicBottleNet(gnet_sizes, output_scale=output_scale.to(device_id))
                                
                self.gnetdict[pname] = GNet(name=pname,
                                            gnet=gnet.to(device_id),
                                            optimizer=optim.Adam(gnet.parameters()),
                                            gnet_input=row_col_encoding.to(device_id),
                                            weights=weight.reshape(param.data.shape),
                                            new_weights=weight.reshape(param.data.shape),
                                            grad_scale=grad_scale)
                                            
    def __repr__(self) -> str:  
        output_str =  f"Gnet parameters:\n"
        for name in self.gnetdict.keys():
            output_str += f"Parameter={name}\n" \
                        f"Parameter shape={self.gnetdict[name].new_weights.shape}\n" \
                        f"GNet input shape={self.gnetdict[name].gnet_input.shape}\n\n"
        return output_str
                                                                         
    def get_num_params_gnet(self) -> int:
        num_params = 0
        for name in self.gnetdict.keys():
            for _, param in self.gnetdict[name].gnet.named_parameters():
                num_params += param.numel()
            
        return num_params
    
    def compression(self, model) -> float:
        num_model_params = sum(p.numel() for p in model.parameters())
        num_gnet_params = self.get_num_params_gnet()
        
        compression = num_model_params/num_gnet_params
        return compression
        
    def correlations(self):
        """
        Returns the correlation between the old and new weights.
        It creates a correlation matrix of size `(len(self.gnet), 1)`
        that captures the correlation between the old and new weights for each
        layer separately. No crosscorrelations!

        Returns:
            np.array: Correlations between old and new weights
        """
        corr_weights = np.ones((len(self.gnet), 1))

        for i, name in enumerate(self.gnetdict.keys()):
            old_weights = np.array(self.gnetdict[name].weights.cpu().data).reshape((1,-1))
            new_weights = np.array(self.gnetdict[name].new_weights.cpu().data).reshape((1,-1))
            cc = np.corrcoef(old_weights, new_weights)
            corr_weights[i] = cc[0, 1]
        return corr_weights
                
    def save(self, fname) -> None:
        print("Saving gnets ...\n") 
        param_dict = {}
            
        for name in self.gnetdict.keys():                                 
            entry_name = name + "_state_dict"
            model_name = "model_" + entry_name
            optimizer_name = "optimizer_" + entry_name
            param_dict[model_name] = self.gnetdict[name].gnet.state_dict()
            param_dict[optimizer_name] = self.gnetdict[name].optimizer.state_dict()
                        
        torch.save(param_dict, fname)
                      
    def load(self, fname) -> None:
        print("Loading gnets ...\n")
        checkpoint = torch.load(fname)
            
        for name in self.gnetdict.keys():                  
            entry_name = name + "_state_dict"
            model_name = "model_" + entry_name
            optimizer_name = "optimizer_" + entry_name                
            self.gnetdict[name].gnet.load_state_dict(checkpoint[model_name])
            self.gnetdict[name].optimizer.load_state_dict(checkpoint[optimizer_name])
         
    def train(self) -> None:
        for name in self.gnetdict.keys():
            self.gnetdict[name].gnet.train()
    
    def zero_grad(self) -> None:
        for name in self.gnetdict.keys():
            self.gnetdict[name].optimizer.zero_grad()
        
    def predict_weights(self, model) -> None:
        """
        This function generates the new weights using the gnets, reshapes them
        to the original shape and sets the models parameters to the corresponding
        new weights.

        Args:
            - `model` (_type_): The neural network model in question.
            - `device` (_type_): The devices where the model is supposed to live.
        """
        for name, param in model.module.named_parameters():
            if name in self.gnetdict.keys():
                gnet_inputs = self.gnetdict[name].gnet_input
                new_weights = self.gnetdict[name].gnet(gnet_inputs)
                new_weights = new_weights.view(param.data.shape)
                self.gnetdict[name].new_weights = new_weights
                # Sets the models parameters to the corresponding new weights
                param.data = nn.Parameter(new_weights)
                
    def mix_weights(self, model, epsilon=1.0):
        """
        This function generates new

        Args:
            - `model` (_type_): The neural network model we want to train.
            - `device` (_type_): The device where we want to train our gnets.
            - `epsilon` (float, optional): _description_. Defaults to 1.0.
        """
        for name, param in model.module.named_parameters():
            if name in self.gnetdict.keys():
                self.gnetdict[name].gnet.eval()
                # Get new weights using gnets
                old_weights = param.data
                self.gnetdict[name].weights = old_weights
                gnet_inputs = self.gnetdict[name].gnet_input
                new_weights = self.gnetdict[name](gnet_inputs)
                new_weights = new_weights.view(param.data.shape)
                self.gnetdict[name].new_weights = new_weights

                # Mix old and new weights  
                new_param = epsilon*old_weights + new_weights*(1.-epsilon)                
                param.data = nn.Parameter(new_param)

    def backward(self, model) -> None:
        """
        This function takes the models gradients after a forward and 
        backward pass through the model and propagates them through the Gnet to
        update the parameters.

        Args:
            - `model` (_type_): The neural network model.
        """
        for name, param in model.module.named_parameters():
            if name in self.gnetdict.keys():
                self.gnetdict[name].weights = param.data
                norm_grad = torch.div(param.grad, self.gnetdict[name].grad_scale)
                self.gnetdict[name].new_weights.backward(norm_grad)
                      
    def step(self) -> None:
        for name in self.gnetdict.keys():
            self.gnetdict[name].optimizer.step()

