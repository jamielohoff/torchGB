from typing import Any, Sequence, Tuple

import torch
import numpy as np


def generate_square_subsequent_mask(sz: int):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_tensor_dimensions(model, layer, input_shape, for_input=False) -> torch.Tensor:
    t_dims = None
    def _local_hook(_, _input, _output):
        nonlocal t_dims
        t_dims = _input[0].size() if for_input else _output.size()
        return _output
    handle = layer.register_forward_hook(_local_hook)
    dummy_var = torch.zeros(*input_shape, dtype=torch.int32).to('cuda')

    bptt = input_shape[0]
    src_mask = generate_square_subsequent_mask(bptt).to("cuda")
    model(dummy_var, src_mask)
    handle.remove()
    return t_dims


def find_layer(model, pname):
    """
    Inverse layer lookup function
    This function iterates through named modules in the model and finds 
    the layer that matches the given parameter name
    """
    found_it = False
    Layer = []
    Mname = []
            
    if not found_it:
        return Layer
    
    # This is awfully slow, but it works
    for mname, layer in model.named_modules():
        for name, param in layer.named_parameters():
            ppname = mname+'.'+name
            if ppname == pname:
                if mname == Mname:
                    Layer = layer
                    return Layer
    
    return Layer


def set_encoding_type(dlist, idx, t):
    if t == 1:        # one-hot vector
        dlist[idx] = idx
    elif t == 2:        # plain binary code
        # b = np.array(list(np.binary_repr(idx).zfill(npbits[i]))).astype(np.int8)
        b = idx
        dlist[idx] = b
    elif t == 3:        # Gray code
        #b = np.array(list(np.binary_repr(idx ^ (idx >> 1)).zfill(npbits[i]))).astype(np.int8)
        b = idx ^ (idx >> 1)
        dlist[idx] = b 
    elif t == 4:        # Linear code
        b = idx
        dlist[idx] = b 
        
    elif t == 5:        # Random code
        dlist[idx] = idx 
    else:
        raise ValueError('Unknown nptype')


def generate_GDN_layer(types: Sequence[int], 
                    dims: Sequence[int], 
                    bits: Sequence[int], 
                    extras: Any = ()) -> Tuple[np.array, np.array]:
    """
    This function creates the inputs and targets used for the training of the
    genomic networks. 
    
    Examples of calling this for 2-layer MNIST

    randomVector = np.random.random_sample((800,3))
    W, GC       = generateGDNlayer((GRY,GRY,RND),[28,28,800],[5,5,3], ((),(),randomVector))
    b, GCbias   = generateGDNlayer((RND),[800],[3],(randomVector,(),()))
    W2, GC2      = generateGDNlayer((RND,HOT),[800,10],[3,10],(randomVector,(),()))
    
    Arguments:
        - types(Sequence[int]): list of types of encoding for each variable
        - dims(Sequence[int]): list of dimensions for each variable
        - bits(Sequence[int]): list of bits for each variable
        - extras(Any): additional information for each variable
        
    Returns:
        - Tuple[np.array, np.array]: Tuple of targets and inputs [W, GC]
    """
    npdims, nptypes, npbits = np.atleast_1d(dims, types, bits)
    
    # Make a row vector so that the shape matches the output
    W1 = np.zeros(npdims.prod()).flatten()      
    W1 = np.expand_dims(W1, 1)                  
    W1 = torch.tensor(W1, dtype=torch.float32)

    GC1 = np.zeros((len(W1), npbits.sum()))
       
    # This will make a list of arrays for each variable    
    var_list = []
    
    for i in range(np.size(nptypes)):
        dlist = np.zeros((npdims[i], 1)).astype(np.int16)
                        
        for idx in range(npdims[i]):
            set_encoding_type(dlist, idx, nptypes[i])
        var_list.append(dlist)
        
    # This will add extra dimensions to the array
    expanded_vars = []
    for i in range(np.size(nptypes)):
        shape = np.ones(npdims.size).astype(np.int16)
        shape[i] = npdims[i]
        
        dlist = np.reshape(var_list[i], shape)
        expanded_vars.append(dlist)
        
    # This will tile singular dimensions        
    expanded_tiled_vars = []

    for i in range(np.size(nptypes)):
        shape = npdims.astype(np.int16)
        # Set the relevant shape to 1
        shape[i] = 1  
        
        # Tile the variables along all dims but one
        dlist = np.tile(expanded_vars[i], shape)   
        # Make a string out of it...WTF why?          
        dlist = np.reshape(dlist,(np.prod(dims), 1), order='F')
        
        if nptypes[i] == 1:                                     # One-hot vector
            max_hot = np.max(dlist)+1
            hots = np.zeros((max_hot,max_hot), dtype=np.int8)
            
            for j in range(max_hot):
                hots[j,j] = 1
                
            hots = hots.astype(np.int8)
            dlist = hots[dlist.squeeze(),:]
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))

        if nptypes[i] == 4:                                     # Linear (non-binary) code
            dlist = dlist #.squeeze()                           # remove singular dimensions 
            
        if (nptypes[i] == 2)|(nptypes[i] == 3):                 # binary code
        
            max_bits = npbits[i]
            max_bins = 2 ** max_bits
            bins = np.zeros((max_bins,max_bits), dtype=np.int8)
            
            # This will be used for binary conversion
            for j in range(max_bins):
                bins[j,:] = np.array(list(np.binary_repr(j).zfill(max_bits))).astype(np.int8)
                
            dlist = bins[dlist.squeeze(),:]                     # convert to binary numbers 
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))
            
            dlist = dlist[:,(max_bits-npbits[i]):(max_bits)]      # take only the lowest bits
            
        if nptypes[i] == 5:                                     # random vectors
            dlist = extras[i][dlist.squeeze(),:];               # add the vector of predefined random numbers
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))
        
        expanded_tiled_vars.append(dlist)
        
    all_vars = expanded_tiled_vars[0]
    
    for i in range(1, np.size(nptypes)):
        all_vars = np.concatenate((expanded_tiled_vars[i], all_vars), axis=1)
        
    # This will detach all GC tensors from the computational graph and thus
    # they will not be differentiated
    GC1 = torch.tensor(all_vars, dtype=torch.float, requires_grad=False)
    
    # Check if rows exceed threshold (for what?)
    for i in range(GC1.shape[1]):
        maxCol = torch.max(GC1[:,i])
        if maxCol > 1e-6:
            GC1[:,i] = GC1[:,i]/maxCol
            
    # Normalize the GC1 so that it has zero mean and unit variance
    with torch.no_grad():
        GC1 = (GC1 - torch.mean(GC1)) / torch.std(GC1) # Xavier
    
    return W1, GC1   

