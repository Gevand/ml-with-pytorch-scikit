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

model = torch.nn.Sequential(
    torch.nn.Linear(2,150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 2),
    torch.nn.ReLU()
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.zero_grad()
model.train()
for step in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_known)
    loss.backward()
    optimizer.step()

model.eval()

class MyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2,150)
        self.fc2 = torch.nn.Linear(150,2)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return x

model = MyNet()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.zero_grad()
model.train()
for step in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_known)
    loss.backward()
    optimizer.step()
model.eval()