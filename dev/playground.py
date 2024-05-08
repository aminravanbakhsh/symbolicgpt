import torch

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    print("MPS is available on this device.")
else:
    print("MPS is not available on this device.")
