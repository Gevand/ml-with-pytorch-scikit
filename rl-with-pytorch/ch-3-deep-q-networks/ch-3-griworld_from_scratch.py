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
    epochs = 1000
    discount_factor = .8
    model = torch.nn.Sequential(torch.nn.Linear(l1, l2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(l2, l3),
                                torch.nn.ReLU(), torch.nn.Linear(l3, l4))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in tqdm(range(epochs)):
        game = Gridworld(size=4, mode='static')
        done = False
        state = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
        state_t = torch.from_numpy(state).float()
        while not done:
            prob = np.random.random()
            move = model(state_t)
            move_index = np.argmax(move.data.numpy())
            if prob < epsilon:
                # generate random move
                move_index = np.random.randint(0, 4)

            move_letter = move_options[move_index]
            game.makeMove(move_letter)
            reward = game.reward()
            if reward != -1:
                done = True
            if done:
                target = torch.tensor(reward, dtype=torch.float)
            else:
                state_next = game.board.render_np().reshape(
                    1, 64) + (np.random.rand(1, 64) / 10.0)
                state_next_t = torch.from_numpy(state_next).float()
                move_next = model(state_next_t)
                max_q = torch.max(move_next)
                target = reward + discount_factor*max_q
                state = state_next
                state_t = state_next_t

            loss = loss_fn(move.squeeze()[move_index],
                           target)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if epsilon > .1:
                epsilon -= (1/epochs)
    return model


def test_model(model: torch.nn.Module, mode='static', display=True):
    game = Gridworld(size=4, mode=mode)
    model.eval()
    done = False
    with torch.no_grad():
        total_reward = 0
        while not done:
            state = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
            state_t = torch.from_numpy(state).detach().float()
            move = model(state_t)
            move_index = np.argmax(move.detach().numpy())
            move_letter = move_options[move_index]
            if display:
                print(game.board.render())
            game.makeMove(move_letter)
            reward = game.reward()
            total_reward += reward
            print("Made move", move_letter, 'and got a reward', reward)
            done = reward != -1


result_model = train_1()
test_model(result_model)
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Loss", fontsize=22)
plt.show()
