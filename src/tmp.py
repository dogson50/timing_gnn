import torch

def change_tensor(t):
    t += 1

a = torch.tensor([1.0])
print(a)
change_tensor(a)
print(a)
