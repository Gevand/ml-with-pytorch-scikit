import torch
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


class ContextBandit:
    def __init__(self, arms: int) -> None:
        self.arms = arms
        self.init_distributions(arms)
        self.update_state()

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def init_distributions(self, arms: int):
        self.bandit_matrix = np.random.rand(arms, arms)

    def reward(self, prob):
        reward = 0
        for _ in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward


arms = 10
N, D_in, H, D_out = 1, arms, 100, arms

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU(),
)
loss_fn = torch.nn.MSELoss()


def one_hot(length, state):
    return_value = [0] * length
    return_value[state] = 1
    return return_value


def softmax(av, tau=1.12):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm


def train(env: ContextBandit, epochs: int = 5000, learning_rate: float = 1e-2):
    model.train()
    cur_state = torch.Tensor(one_hot(arms, env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state)
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
        av_softmax /= av_softmax.sum()
        choice = np.random.choice(arms, p=av_softmax)
        cur_reward = env.choose_arm(choice)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms, env.get_state()))
        if (i % 250 == 0):
            print(f'At epoch {i} - last reward was {reward}')
    return np.array(rewards, dtype=np.float64)


env = ContextBandit(arms)
rewards = train(env)
mean_rewards = rewards.cumsum()
for i in range(len(mean_rewards)):
    mean_rewards[i] /= i+1
fig, ax = plt.subplots(1, 1)
ax.scatter(np.arange(len(rewards)), mean_rewards)
ax.set_xlabel("Neural network rewards over epochs")
plt.show()
