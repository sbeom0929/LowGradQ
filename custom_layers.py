import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function     
from torch.nn.parameter import Parameter 
from torch.nn import init                
from torch.nn import Module              
from typing import Optional, List, Tuple, Union
import math
from torch import Tensor

from quantizer import * 

class custom_Conv2d_function(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, X, weight, bias = True, stride = 1, padding = 0, dilation = 1, groups = 1):    
     
        ctx.save_for_backward(X, weight, bias) # To perform the backward process, save the activation, weight, and bias during the forward process.
        ctx.stride = stride 
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Forward quantization
        act_dequant = act_quantization(X,8)
        weight_dequant = weight_quantization(weight,8)

        out = F.conv2d(act_dequant, weight_dequant, bias, stride, padding, dilation, groups)

        return out 
    
    @staticmethod
    def backward(ctx, grad_output):     # grad_output : local gradient(loss map)
        grad_X = grad_w = grad_b = None
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups

        act_dequant = act_quantization(input,8)
        weight_dequant = weight_quantization(weight,8)        
        
        # Optimal Threshold Search (OTS) process
        optim_threshold, max_abs_value = find_optim_threshold(grad_output)
        # Dual-Scale Adaptive Gradient Quantization
        grad_dequant = grad_quantization(grad_output, 8, optim_threshold, max_abs_value) 
        
        if ctx.needs_input_grad[0]: 
            grad_X = torch.nn.grad.conv2d_input(input.shape, weight_dequant, grad_dequant, stride, padding, dilation, groups)  
     
        if ctx.needs_input_grad[1]:  
            grad_w = torch.nn.grad.conv2d_weight(act_dequant, weight.shape, grad_dequant, stride, padding, dilation, groups)   
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = grad_output.sum(dim=(0,2,3))    
        
        return grad_X, grad_w, grad_b, None, None, None, None       

class custom_Conv2d(Module):   
    __constants__ = ['stride', 'padding', 'dilation', 'groups']    
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = True, groups = 1, device = None, dtype = None) -> None: 
        super(custom_Conv2d, self).__init__()

        factory_kwargs = {'device': device, 'dtype' : dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        self.groups = groups
        self.weight = Parameter(torch.empty(        
                (out_channels, in_channels//groups, kernel_size, kernel_size), **factory_kwargs)   
                )   

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)  

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init.calculate_fan_in_and_fan_out(self.weight)
            #fan_in, _ = calculate_fan_in_and_fan_out(self.weight) 
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor: 
        return custom_Conv2d_function.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class custom_Linear_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)

        output = input.matmul(weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class custom_Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(custom_Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return custom_Linear_function.apply(input, self.weight, self.bias)

def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out