import gym
env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.reset())