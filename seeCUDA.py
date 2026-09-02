import torch

# Print PyTorch version
print(torch.__version__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA is available: {cuda_available}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Get the CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
else:
    print("CUDA is not available on your system.")
