import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

total = 0
for _ in range(1000):
    action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)
    total += reward

    if terminated or truncated:
        observation, info = env.reset()
        print(f'episode return = {total}')
        total = 0

env.close()