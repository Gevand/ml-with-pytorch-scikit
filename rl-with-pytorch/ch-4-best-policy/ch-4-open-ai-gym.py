import gym

env = gym.make('CartPole-v1',render_mode="human")

total_game_count = 0
state1 = env.reset()
while total_game_count < 50:
    action = env.action_space.sample()
    state, reward, done, trunc, info = env.step(action)
    if done or trunc:
        total_game_count = total_game_count + 1
        state1 = env.reset()
    env.render()
