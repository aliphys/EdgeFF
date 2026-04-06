import torch
import torch_tensorrt as ttr 


print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
try:
    print("torch_tensorrt:", ttr.__version__)
except Exception as e:
   print("torch_tensorrt: not available:", e)