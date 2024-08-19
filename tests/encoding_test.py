import unittest
import numpy as np
import torch

from torchGB.utils import make_row_col_encoding, EncodingType




########################

def set_encoding_type(dlist, idx, t) -> None:
    """
    TODO docstring
    TODO refactor this
    Args:
        `dlist` (Sequence[int]): _description_
        `idx` (int): _description_
        `t` (int): _description_
    Raises:
        ValueError: _description_
    """
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
        raise ValueError("Unknown nptype")
def old_make_row_col_encoding(types, dims, bits, extras = ()):
    """
    TODO refactor this
    This function creates the inputs and targets used for the training of the
    genomic networks. 
    
    Examples of calling this for 2-layer MNIST
    randomVector = np.random.random_sample((800,3))
    W, GC       = generateGDNlayer((GRY,GRY,RND),[28,28,800],[5,5,3], ((),(),randomVector))
    b, GCbias   = generateGDNlayer((RND),[800],[3],(randomVector,(),()))
    W2, GC2      = generateGDNlayer((RND,HOT),[800,10],[3,10],(randomVector,(),()))
    
    Arguments:
        `types` (Sequence[int]): list of types of encoding for each variable
        `dims` (Sequence[int]): list of dimensions for each variable
        `bits` (Sequence[int]): list of bits for each variable
        `extras` (Any): additional information for each variable
        
    Returns:
        - Tuple[np.array, np.array]: Tuple of targets and inputs [W, GC]
    """
    npdims, nptypes, npbits = np.atleast_1d(dims, types, bits)
    
    # Make a row vector so that the shape matches the output
    weight = np.zeros(npdims.prod()).flatten()      
    weight = np.expand_dims(weight, 1)                  
    weight = torch.tensor(weight, dtype=torch.float32)
    row_col_encoding = np.zeros((npdims.prod(), npbits.sum()))
       
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
    row_col_encoding = torch.tensor(all_vars, dtype=torch.float, requires_grad=False)
    
    # Check if rows exceed threshold (for what?)
    for i in range(row_col_encoding.shape[1]):
        maxCol = torch.max(row_col_encoding[:,i])
        if maxCol > 1e-6:
            row_col_encoding[:,i] = row_col_encoding[:,i]/maxCol
            
    # Normalize the row_col_encoding so that it has zero mean and unit variance
    with torch.no_grad():
        row_col_encoding = (row_col_encoding - torch.mean(row_col_encoding)) / torch.std(row_col_encoding) # Xavier
        row_col_encoding -= row_col_encoding.min() # inputs larger than 0
    return row_col_encoding    


class TestBinaryEncoding(unittest.TestCase):
    def test_encoding(self):
        encoding_types = (EncodingType.BINARY, EncodingType.BINARY)
        shape = (4, 4)
        
        num_encoding_bits = np.ceil(np.log(shape)/np.log(2)).astype(np.uint16)
        num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
        
        encoding = make_row_col_encoding(shape, encoding_types, num_encoding_bits)
        true_encoding = old_make_row_col_encoding((2, 2), (4, 4), num_encoding_bits)
        print(encoding)
        print(true_encoding)
        self.assertTrue((true_encoding-encoding).sum() < 1e-7)


if __name__ == "__main__":
    unittest.main()

