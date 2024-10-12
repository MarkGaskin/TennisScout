import torch
print(torch.__version__)               # Should display PyTorch version
print(torch.version.cuda)              # Should display CUDA version (e.g., '11.8')
print(torch.cuda.is_available())       # Should return True
print(torch.cuda.get_device_name(0))   # Should display 'GeForce GTX 1070'
