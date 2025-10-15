import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

run_name = "test1"
model_path = f'./models/{run_name}/dqn_cartpole'
config_file_path = f'./logs/{run_name}'

# set up an environment
env = gym.make("CartPole-v1", render_mode=None) # "human"

# create a DQN agent (use sb3 implementation)
model = DQN("MlpPolicy", env, verbose=0)

# create a logger to collect data
mylogger = configure(config_file_path, ["csv", "tensorboard"])
model.set_logger(mylogger)

# train and save the model (logging is happening during the training)
model.learn(total_timesteps=10_000)
model.save(model_path)