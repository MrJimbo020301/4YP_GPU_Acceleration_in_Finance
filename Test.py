import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
print("Number of GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
