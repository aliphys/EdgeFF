#!/usr/bin/env python3

import torch

# Test GPU availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    
    # Test basic operations
    device = torch.device('cuda')
    x = torch.randn(5, 3).to(device)
    y = torch.randn(3, 4).to(device)
    z = torch.mm(x, y)
    print("GPU tensor operation successful!")
    print("Result device:", z.device)
    print("Result shape:", z.shape)
else:
    print("CUDA not available, will use CPU")
