import os

# Check if PYTORCH_CUDA_ALLOC_CONF is set
pytorch_cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF")

if pytorch_cuda_alloc_conf is not None:
    print(f"PYTORCH_CUDA_ALLOC_CONF is set to: {pytorch_cuda_alloc_conf}")
else:
    print("PYTORCH_CUDA_ALLOC_CONF is not set.")

import torch 

# x = posterior
# y = deterministic 
# output_shape = self.observation_shape
# input_shape=(-1,)