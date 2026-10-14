import torch

dimensions = 3

w = torch.nn.Parameter(torch.ones(1, dimensions))

print("Trọng số:", w)
print("Kích thước:", w.shape)
print("Là Parameter:", isinstance(w, torch.nn.Parameter))