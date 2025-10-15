import os
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

# Create log directory
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment with Monitor
env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)

# Train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Plot the results
plot_results([log_dir], 10_000, results_plotter.X_TIMESTEPS, "PPO CartPole")
plt.savefig("result.pdf", format="pdf")
plt.show()