# LowGradQ: Adaptive Gradient Quantization for Low-Bit CNN Training via Kernel Density Estimation-Guided Thresholding and Hardware-Efficient Stochastic Rounding Unit

<p align="center">
  <img src="Figures/Overall%20Quantization%20Framework.jpg" alt="Overall Quantization Framework">
</p>


## Abstract
This paper proposes a hardware-efficient INT8 training framework with dual-scale adaptive gradient quantization (DAGQ) to cope with the growing need for efficient on-device CNN training. DAGQ captures both small- and large-magnitude
gradients, ensuring robust low-bit training with minimal quantization error. Additionally, to reduce the computational and memory
demands of stochastic rounding in low-bit training, we introduce a reusable LFSR-based stochastic rounding unit (RLSRU), which
efficiently generates and reuses random numbers, minimizing hardware complexity. The proposed framework achieves stable INT8 training across various networks with minimal accuracy loss while being implementable on RTL-based hardware accelerators, making it well-suited for resource-constrained environments.

## Getting Started
### Environment
Python: 3.8.0

Pytorch 2.2.1

### Running the Experiment
To run the experiment, use the following command:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --pretrained=pretrained_weight/resnet/resnet18-5c106cde.pth --save-dir=results/resnet18_quant --lr=0.001
```

### Citation
If you find this repo useful in your research, please consider citing the following paper:
