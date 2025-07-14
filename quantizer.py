import torch 
import math
from typing import Tuple, List

######## Optimal Threshold Search (OTS) Process
def find_optim_threshold(x):
    max_abs_value = torch.max(torch.abs(x))
    grad_mean = x.mean()
    grad_std = x.std()

    lower_threshold = 0.0280 * grad_std     # 0.0280(min mf), 0.0601(max mf) is the magnitude factor (mf), which may be chosen to have a different value depending on the network.
    upper_threshold = 0.0601 * grad_std

    search_precision = 5
    thresholds = torch.linspace(lower_threshold, upper_threshold, search_precision, device=x.device)
    
    def compute_mse(threshold):
        threshold_above_value = x[torch.abs(x) > threshold]
        dequant_threshold_above_value = grad_quantization(threshold_above_value, 8, threshold, max_abs_value)
        return torch.mean((threshold_above_value - dequant_threshold_above_value) ** 2)

    mse_errors = torch.tensor([compute_mse(threshold) for threshold in thresholds], device=x.device)
    min_mse_index = torch.argmin(mse_errors)
    optim_threshold = thresholds[min_mse_index]

    return optim_threshold, max_abs_value

@torch.jit.script
def stochastic_rounding(x: torch.Tensor) -> torch.Tensor:
    error_prob = x - torch.floor(x)
    random = torch.rand_like(x)
    rounded = torch.where(random < error_prob, torch.ceil(x), torch.floor(x))
    return rounded

@torch.jit.script   
def grad_quantization(x: torch.Tensor, bits: int, threshold: torch.Tensor, max_abs_value: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    quant_level = (2 ** (bits - 1)) - 1

    scale_factor1 = threshold / quant_level
    scale_factor2 = (max_abs_value - threshold) / quant_level

    grad_quant = torch.where(torch.abs(x) <= threshold, x / scale_factor1,
                             torch.where(x < -threshold, (x + threshold) / scale_factor2, (x - threshold) / scale_factor2))

    condition_tensor = torch.where(torch.abs(x) <= threshold,  
                                   torch.tensor(0, device=x.device),
                                   torch.where(x < -threshold, torch.tensor(1, device=x.device), torch.tensor(2, device=x.device)))

    grad_quant = stochastic_rounding(grad_quant)
    grad_quant = torch.clamp(grad_quant, -127, 127)

    grad_dequant = torch.where(condition_tensor == 0, grad_quant * scale_factor1,
                               torch.where(condition_tensor == 1, (grad_quant * scale_factor2) - threshold,
                                           (grad_quant * scale_factor2) + threshold))

    return grad_dequant

def weight_quantization(x,bits):
    max_abs_value = torch.max(torch.abs(x))
    bits = math.pow(2, bits - 1) - 1 # symmetric quantization 
    scale_factor = max_abs_value / bits  

    weight_quant = torch.round(x / scale_factor) 
    wieght_quant = torch.clamp(weight_quant , -bits, bits)
    weight_dequant = weight_quant * scale_factor

    return weight_dequant 

def act_quantization(x,bits):
    max_abs_value = torch.max(torch.abs(x))
    bits = math.pow(2, bits - 1) - 1 # symmetric quantization 
    scale_factor = max_abs_value / bits  

    act_quant = torch.round(x / scale_factor) 
    act_quant = torch.clamp(act_quant, -bits, bits)
    act_dequant = act_quant * scale_factor
    
    return act_dequant 
