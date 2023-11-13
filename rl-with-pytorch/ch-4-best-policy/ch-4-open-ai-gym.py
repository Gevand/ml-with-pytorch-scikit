import gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return


l1 = 4  # <- input layer
l2 = 150  # <- middle layer
l3 = 2  # <- output

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))


MAX_DUR = 200
MAX_EPISODES = 500
gamma = 0.99
score = []

env = gym.make('CartPole-v1', render_mode="rgb")

total_game_count = 0
state1 = env.reset()
for episode in tqdm(range(MAX_EPISODES)):
    curr_state = env.reset()[0]
    done = False
    transitions = []

    for t in range(MAX_DUR):
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state, _, done, _, info = env.step(action)
        transitions.append((prev_state, action, t+1))
        if done:
            break

    ep_len = len(transitions)
    score.append(ep_len)
    reward_batch = torch.Tensor(
        [r for (s, a, r) in transitions]).flip(dims=(0,))
    disc_rewards = discount_rewards(reward_batch)
    state_batch = torch.Tensor(
        [s for (s, a, r) in transitions])
    action_batch = torch.Tensor(
        [a for (s, a, r) in transitions])
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(
        dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    loss = loss_fn(prob_batch, disc_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(score)
avg_score = running_mean(score, 50)
plt.plot(avg_score)
plt.show()
