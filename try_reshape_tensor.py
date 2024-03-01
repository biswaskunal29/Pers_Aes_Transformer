import torch



a = torch.tensor([[0, 1, 2], [3, 4, 5]])
print(a)
b = torch.reshape(a, (-1,))
print(b)
c = torch.reshape(a, (2, 3))
print(c)

f = torch.flatten(a)
print(f)
g = torch.reshape(f, (2, 3))
print(g)














