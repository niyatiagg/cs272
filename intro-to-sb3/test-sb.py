import gymnasium as gym
from stable_baselines3 import DQN

run_name = "test"
model_path = f'./models/{run_name}/dqn_cartpole'

# set up an environment
env = gym.make("CartPole-v1", render_mode=None) # "human"

# load the model
model = DQN.load(model_path)

# run the trained model for performance test (similar to ML inference)
ret = 0.0
obs, info = env.reset() # initialize the state
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ret += reward
    if terminated or truncated:
        print(f'episode return: {ret}')
        ret = 0.0
        obs, info = env.reset()