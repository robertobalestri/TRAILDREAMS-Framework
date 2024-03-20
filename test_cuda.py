import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")
    return device

device = check_cuda()
print("Using device:", device)