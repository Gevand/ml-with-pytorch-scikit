import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
n = 10
probs = np.random.rand(n) #10 random probabilities for each arm
eps = .2

print(probs)

def get_reward(prob, n=10):
    reward = 0
    for i in range(n):
        if random.random() < prob:
            reward +=1
    return reward

print(np.mean([get_reward(0.7) for _ in range(2000)]))


def update_record(record,action,r):
    #compute the new average, at 0 index is the number of actions, at 1 index is the average 
    #current average * number of actions gets you the old total reward, add the new reward and divide by the number of actions + 1 to get the new average
    new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1)
    record[action,0] += 1
    record[action,1] = new_r
    return record

def get_best_arm(record):
    arm_index = np.argmax(record[:,1],axis=0)
    return arm_index

record = np.zeros((n,2))
rewards = [0]
for i in range(500):
    if random.random() > eps:
        choice = get_best_arm(record)
    else:
        choice = np.random.randint(10)
    r = get_reward(probs[choice])
    record = update_record(record, choice,r)
    mean_reward = ((i + 1) * rewards[-1] + r)/ (i+2)
    rewards.append(mean_reward)

print(rewards)
fig, ax = plt.subplots(1,2)
ax[0].scatter(np.arange(len(rewards)),rewards)
ax[0].set_xlabel("Plays - Epsilon Greedy")

def softmax(av, tau=1.12):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm

x = np.arange(10)
print(x)
av = np.zeros(10)
p = softmax(av)
print(p) #since the av is all 0s, the distribution is uniform

record = np.zeros((n,2))
rewards = [0]
for i in range(500):
    p = softmax(record[:,1])
    choice = np.random.choice(np.array(n), p=p)
    r = get_reward(probs[choice])
    record = update_record(record, choice,r)
    mean_reward = ((i + 1) * rewards[-1] + r)/ (i+2)
    rewards.append(mean_reward)

print(rewards)
ax[1].scatter(np.arange(len(rewards)),rewards)
ax[1].set_xlabel("Plays - Softmax")
plt.show()

class ContextBandit:
    def __init__(self, arms:int ) -> None:
        self.arms = arms
        self.init_distributions(arms)
        self.update_state()
    
    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def init_distributions(self, arms:int):
        self.bandit_matrix = np.random.rand(arms,arms)
    
    def reward(self, prob):
        reward = 0
        for _ in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state (self):
        return self.state
    
    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])
    
    def choose_arm(self,arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward

env = ContextBandit(arms=10)
state = env.get_state()
reward = env.choose_arm(1)
print(state, 'with reward:', reward)