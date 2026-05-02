
import gymnasium as gym
import numpy as np  
import torch
from neuralnet import PolicyNet
from torch.distributions import Categorical


env = gym.make("CartPole-v1", render_mode="human")
model = PolicyNet()
model.load_state_dict(torch.load("a.pth"))
# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
total_reward = 0
for _ in range(1000):
    # this is where you would insert your policy
    with torch.no_grad():                                                    
          logits = model(torch.as_tensor(observation, dtype=torch.float32))
          action = Categorical(logits=logits).sample().item()
    
   #  print(f"action = {action}")

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode finished! Total reward: {total_reward}")
        total_reward = 0
    

env.close()