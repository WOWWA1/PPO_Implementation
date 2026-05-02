import gymnasium as gym 
import numpy as np

epochs = 100 #how many policy upgrades should we do
N = 200 #how many trajectories should we collect for each V fitting

for i in range(epochs):
    