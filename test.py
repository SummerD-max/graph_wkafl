import torch
x = torch.randn(3, 4)
print(x)
res, indices = torch.sort(x)

print(res)
print(indices)

res, indices = torch.sort(x, 0)
print(res)
print(indices)
x = torch.tensor([0, 1] * 9)
x.sort()
x.sort(stable=True)