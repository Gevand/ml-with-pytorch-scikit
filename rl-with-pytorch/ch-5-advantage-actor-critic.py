import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as f
import gym

import torch.multiprocessing as mp


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = f.normalize(x, dim=0)
        y = f.relu(self.l1(x))
        y = f.relu(self.l2(y))
        actor = f.log_softmax(self.actor_lin1(y), dim=0)
        # detach removes the tensor fromt he coimputation graph, so it won't require a gradient
        c = f.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic



def worker(t, worker_model, counter, params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, citic_loss, eplen = update_params(
            worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = f.normalize(Returns, dim=0)
    actor_loss = -1 * logprobs * (ret_ - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)


def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    while (done == False):
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info, _ = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards

MasterNode = ActorCritic()
MasterNode.share_memory()
processes = []
params = {
    'epochs': 10,
    'n_workers': 7,
}
counter = mp.Value('i', 0)
for i in range(params['n_workers']):
    p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
for p in processes:
    p.terminate()
print('done')