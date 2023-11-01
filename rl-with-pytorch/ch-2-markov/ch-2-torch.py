import torch
import numpy as np

np_array = np.array([[1,2,3],[4,5,6]])
print(np_array)

torch_tensor = torch.Tensor([[1,2,3],[4,5,6]])
print(torch_tensor)

x = torch.Tensor([2,4])
m = torch.randn(2, requires_grad=True)
b = torch.randn(1, requires_grad=True)
y = m*x + b
y_known = torch.Tensor([1,1])
loss = (torch.sum(y_known - y))**2
loss.backward()
print(m.grad)