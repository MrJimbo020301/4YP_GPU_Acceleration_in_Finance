import torch
print(torch.__version__)

import torch
print(torch.version.cuda)         # CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Checks if a CUDA GPU is accessible
print(torch.cuda.get_device_name(0))  # If CUDA is available, get GPU name
