from collections import deque
from gridworld import Gridworld
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

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


def train1():  # overfit for the first poisition, won't work well with random initializations of the board
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


epochs = 5000
losses = []
mem_size = 1000
batch_size = 200


def train2():  # has the build in "remember" queue, which helps with random states
    replay = deque(maxlen=mem_size)
    max_moves = 50
    h = 0
    for i in tqdm(range(epochs)):
        state_ = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
        state1 = torch.from_numpy(state_).float()
        status = 1
        mov = 0
        while (status == 1):
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(
                1, 64) + (np.random.rand(1, 64) / 10.0)
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat(
                    [s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor(
                    [a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor(
                    [r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat(
                    [s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor(
                    [d for (s1, a, r, s2, d) in minibatch])
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                Y = reward_batch + gamma * \
                    ((1 - done_batch) * torch.max(Q2, dim=1)
                     [0])  # when done = 1, it zeroes out the 1- donebatch expression, so Y becomes just "reward"
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0


def train3():  # replay + a second network that is updated every 500 steps
    model = torch.nn.Sequential(torch.nn.Linear(l1, l2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(l2, l3),
                                torch.nn.ReLU(), torch.nn.Linear(l3, l4))
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    sync_freq = 500
    sync_count = 0
    loss_fn = torch.nn.MSELoss()
    learning_rate = .001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay = deque(maxlen=mem_size)
    max_moves = 50
    h = 0
    for i in tqdm(range(epochs)):
        state_ = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
        state1 = torch.from_numpy(state_).float()
        status = 1
        mov = 0
        while (status == 1):
            sync_count += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(
                1, 64) + (np.random.rand(1, 64) / 10.0)
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward > 0 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat(
                    [s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor(
                    [a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor(
                    [r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat(
                    [s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor(
                    [d for (s1, a, r, s2, d) in minibatch])
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)
                Y = reward_batch + gamma * \
                    ((1 - done_batch) * torch.max(Q2, dim=1)
                     [0])  # when done = 1, it zeroes out the 1- donebatch expression, so Y becomes just "reward"
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if sync_count % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0


train3()
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.show()
