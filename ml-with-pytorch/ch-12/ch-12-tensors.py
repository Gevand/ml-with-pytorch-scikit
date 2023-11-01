import torch
import numpy as np

np.set_printoptions(precision=3)
a = [1,2,3]
b = np.array([4,5,6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)

print(t_a)
print(t_b)

rand_tensor = torch.rand(2,3)
print(rand_tensor)

t_a_new = t_a.to(torch.int64)
print(t_a_new, t_a_new.dtype)

t = torch.rand(3,5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, ' --> ', t_tr.shape)

t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t, 'vs', t_reshape)

t = torch.zeros(1,2,1,4,1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, ' --> ', t_sqz.shape)

torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))

t3 = torch.multiply(t1, t2)
print(t1)
print('mult')
print(t2)
print('=')
print(t3)

t4 = torch.mean(t1, axis=0)
print(t1)
print('vs')
print(t4)

t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)

t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
print(t6)

norm_t1 = torch.linalg.norm(t1, ord=2,dim=1)
print(norm_t1)


torch.manual_seed(1)
t = torch.rand(6)
print(t)
t_splits = torch.chunk(t,3)
[print(item.numpy()) for item in t_splits]
t = torch.rand(5)
print(t)
t_splits = torch.split(t, split_size_or_sections=[3, 2])
[print(item.numpy()) for item in t_splits]

A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A, B], axis=0)
print(C)

A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)
print(S)