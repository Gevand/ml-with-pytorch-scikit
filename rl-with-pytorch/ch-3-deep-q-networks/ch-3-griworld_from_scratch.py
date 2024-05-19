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


def train_1():
    print("Training a simple rl model")
    epsilon = 1
    discount_factor = .8
    model = torch.nn.Sequential(torch.nn.Linear(l1, l2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(l2, l3),
                                torch.nn.ReLU(), torch.nn.Linear(l3, l4))

    for epoch in range(1000):
        game = Gridworld()
        done = False
        state = game.board.render_np().reshape(1, 64) + (np.random.rand(1, 64) / 10.0)
        state_t = torch.from_numpy(state).detach().float()
        while not done:

            prob = np.random.random()
            move = -1

            move = model(state_t).detach().numpy()
            if prob < epsilon:
                # generate random move
                move = np.random.random(4)

            move_index = np.argmax(move)
            move_letter = move_options[move_index]
            game.makeMove(move_letter)
            reward = game.reward()
            if reward != -1:
                done = True
            if done:
                target = move
                target[move_index] = reward
            else:
                state_next = game.board.render_np().reshape(
                    1, 64) + (np.random.rand(1, 64) / 10.0)
                state_next_t = torch.from_numpy(state).detach().float()
                move = model(state_next_t).detach().numpy()
                max_q = move
                target = reward + discount_factor*max_q
                state = state_next
                state_t = state_next_t

            epsilon = max(epsilon - .01, 0)
            # getting a 10 or -10 means we are done


train_1()
