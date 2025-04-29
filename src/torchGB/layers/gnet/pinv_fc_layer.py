import torch
import torch.nn as nn
from torch.autograd import Function
import math

# 1. Define the Custom Autograd Function
class PseudoInverseLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        Forward pass: Standard linear transformation.
        y = x @ weight.T + bias
        """
        # Save tensors needed for backward:
        # - x: needed for dL/dW
        # - weight: needed for dL/dX (via pinv) and standard dL/dX if comparing
        # We don"t strictly need bias for gradient calculation w.r.t. inputs/weights
        ctx.save_for_backward(x, weight)

        # Standard forward computation
        output = x @ weight.t()
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients w.r.t. inputs of forward.
        grad_output is dL/dY
        """
        # NOTE: This implementation is incorrect
        x, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None # Initialize gradients

        # --- Custom Gradient Calculation for Input (dL/dX) ---
        # Standard would be: grad_input = grad_output @ weight
        # Custom: grad_input = grad_output @ pinv(weight)
        if ctx.needs_input_grad[0]: # Check if grad w.r.t x is required
            grad_input = grad_output @ weight


        # --- Standard Gradient Calculation for Weight (dL/dW) ---
        if ctx.needs_input_grad[1]: # Check if grad w.r.t weight is required
            # dL/dW = (dL/dY).T @ X
            try:
                x_pinv = torch.linalg.pinv(x.t())
                grad_weight = grad_output @ x_pinv
            except torch.linalg.LinAlgError as e:
                print(f"Warning: Pseudo-inverse computation failed: {e}")
                # Fallback or raise error? Using zeros as a fallback example.
                grad_weight = torch.zeros_like(weight)

        # --- Standard Gradient Calculation for Bias (dL/dB) ---
        # Bias gradient exists only if bias was provided in forward
        if ctx.needs_input_grad[2]: # Check if grad w.r.t bias is required
             # dL/dB = sum(dL/dY, axis=0)
            grad_bias = grad_output.sum(0)

        # Return gradients in the *exact* order of forward inputs (x, weight, bias)
        return grad_input, grad_weight, grad_bias

# 2. Wrap in an nn.Module
class PseudoInverseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            # Register bias as None if not used - important for autograd function signature
            self.register_parameter("bias", None)

        # Initialize parameters (same as standard nn.Linear for consistency)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom autograd function"s apply method
        return PseudoInverseLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
