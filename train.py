import gymnasium as gym 
import numpy as np
from neuralnet import PolicyNet
from neuralnet import ValueNet
# HYPERPARAMETERS
epochs = 100 #how many policy upgrades should we do
N = 200 #how many trajectories should we collect for each V fitting
model = PolicyNet()
for i in range(epochs):
    
    env = gym.make("CartPole-v1", render_mode="human")


    observation, info = env.reset()

    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()