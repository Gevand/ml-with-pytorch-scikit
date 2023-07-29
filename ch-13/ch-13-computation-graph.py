import torch.nn as nn
import torch


def compute_z(a, b, c):
    # z = 2 × (a – b) + c
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z


print('Scalar Inputs:', compute_z(
    torch.tensor(1), torch.tensor(2), torch.tensor(3)))
print('Rank 1:', compute_z(torch.tensor(
    [1]), torch.tensor([2]), torch.tensor([3])))
print('Rank 2:', compute_z(torch.tensor(
    [[1]]), torch.tensor([[2]]), torch.tensor([[3]])))

a = torch.tensor(3.14, requires_grad=True)
print(a)
b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(b)
w = torch.tensor([1.0, 2.0, 3.0])
print('Requires grad is set to false by default  ', w, w.requires_grad)
print(w.requires_grad_())


torch.manual_seed(1)
w = torch.empty(2, 3)
nn.init.xavier_normal_(w)
print(w)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)

    def forward():
        pass


# simple example of gradient computation, z = wx + b
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
# z = wx + b
z = torch.add(torch.mul(w, x), b)
# mse
loss = (y-z).pow(2).sum()
loss.backward()
print('dL/dw ', w.grad)
print('dL/db', b.grad)

# dL/dw should be 2𝑥(𝑤𝑥 + 𝑏 - y)\
print(2 * x * ((w * x + b) - y), ' vs ', w.grad)

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(),
                      nn.Linear(16, 32), nn.ReLU())
nn.init.xavier_uniform_(model[0].weight)
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
