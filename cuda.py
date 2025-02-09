import torch

if torch.cuda.is_available():
    print("CUDA è disponibile! 🚀")
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA non è disponibile. ❌")

