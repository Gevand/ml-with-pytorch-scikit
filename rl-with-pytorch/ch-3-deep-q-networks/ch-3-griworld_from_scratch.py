from collections import deque
import math
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

move_options = ['u', 'd', 'l', 'r']
losses = []


def train_1():
    print("Training a simple rl model")
    epsilon = 1
    discount_factor = .8
    model = torch.nn.Sequential(torch.nn.Linear(l1, l2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(l2, l3),
                                torch.nn.ReLU(), torch.nn.Linear(l3, l4))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in tqdm(range(1000)):
        game = Gridworld(size=4, mode='static')
        done = False
        state = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
        state_t = torch.from_numpy(state).detach().float()
        while not done:

            prob = np.random.random()
            move = -1

            move = model(state_t)
            move_index = np.argmax(move.detach().numpy())
            if prob < epsilon:
                # generate random move
                move_index = np.random.randint(0, 4)

            move_letter = move_options[move_index]
            game.makeMove(move_letter)
            reward = game.reward()
            if reward != -1:
                done = True
            if done:
                target = move
                target = target.squeeze()
                target[move_index] = reward
            else:
                state_next = game.board.render_np().reshape(
                    1, 64) + (np.random.rand(1, 64) / 10.0)
                state_next_t = torch.from_numpy(state).detach().float()
                move_next = model(state_next_t).detach()
                max_q = move_next
                target = reward + discount_factor*max_q
                state = state_next
                state_t = state_next_t

            # target and move are shape 4 tensors, the original code is shape 1, and it only sends in the move that was made, ignoring the moves that weren't picked
            loss = loss_fn(move.squeeze(),
                           target.squeeze())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            epsilon = max(epsilon - .01, 0)
            # getting a 10 or -10 means we are done


train_1()
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.show()
