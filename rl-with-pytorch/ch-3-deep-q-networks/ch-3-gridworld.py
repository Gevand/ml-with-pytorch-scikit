from gridworld import Gridworld
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
game = Gridworld()
print(game.display())
game.makeMove('d')
game.makeMove('d')
game.makeMove('l')
print(game.display())
print(game.board.render_np())

l1 = 64
l2 = 150
l3 = 100
l4 = 4


model = torch.nn.Sequential(torch.nn.Linear(l1, l2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(l2, l3),
                            torch.nn.ReLU(), torch.nn.Linear(l3, l4))
loss_fn = torch.nn.MSELoss()
learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

gamma = .9
epsilon = 1.0

epochs = 1000
losses = []
for i in tqdm(range(epochs)):
    game = Gridworld(size=4, mode='static')
    # game is a 4 x 4 with 4 arrays signifying the 4 possible items on the screen  so it gets reshaped to 1 x 64 + some random noise is added
    state_ = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
    state1 = torch.from_numpy(state_).float()
    status = 1
    while (status == 1):
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action=action)
        state2_ = game.board.render_np().reshape(
            1, 64) + (np.random.rand(1, 64) / 10.0)
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        with torch.no_grad():
            newQ = model(state2.reshape(1, 64))
        maxQ = torch.max(newQ)
        if reward == -1:
            Y = reward + (gamma * maxQ)
        else:
            Y = torch.tensor(reward, dtype=torch.float)

        Y = Y.detach()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state1 = state2
        if reward != -1:
            status = 0
    if epsilon > .1:
        epsilon -= (1/epochs)

plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.show()
