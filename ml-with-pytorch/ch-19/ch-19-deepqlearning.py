import gym
import torch
from dqn_agent import DQNAgent, Transition
import numpy as np

np.random.seed(1)

# General settings
EPISODES = 200
batch_size = 32
init_replay_memory_size = 500
if __name__ == '__main__':
    env = gym.make('CartPole-v1',render_mode="human")
    agent = DQNAgent(env)
    state = env.reset()
    state = np.asarray(state[0:1])
    # Filling up the replay-memory
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state,
                                [1, agent.state_size])
        agent.remember(Transition(state, action, reward,
                                  next_state, done))
        if done:
            state = env.reset()
            state = np.asarray(state[0:1])
        else:
            state = next_state

    total_rewards, losses = [], []
    for e in range(EPISODES):
        state = env.reset()
        if e % 10 == 0:
            env.render()
        state = np.asarray(state[0:1])
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state,
                                    [1, agent.state_size])
            agent.remember(Transition(state, action, reward,
                                      next_state, done))
            state = next_state
            if e % 10 == 0:
                env.render()
            if done:
                total_rewards.append(i)
                print(f'Episode: {e}/{EPISODES}, Total reward: {i}')
                break
            loss = agent.replay(batch_size)
            losses.append(loss)
