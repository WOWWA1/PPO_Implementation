import gymnasium as gym 
import numpy as np
from neuralnet import PolicyNet
from neuralnet import ValueNet
from torch.distributions import Categorical
env = gym.make("CartPole-v1")
# HYPERPARAMETERS
epochs = 100 #how many policy upgrades should we do
N = 200 #how many trajectories should we collect for each V fitting
gamma = 0.99 #discount factor
lr = 1e-3 #learning rate
eps = 0.2 #clipping range
model = PolicyNet()
V = ValueNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(V.parameters(), lr = lr)

for i in range(epochs):
    observations = []
    rewards = []
    actions = []
    log_probs = []
    lengths = []     
    for j in range(N):
        observation, info = env.reset()
        total_reward = 0
        episode_over = False
        t = 0
        while not episode_over:
            t += 1
            observations.append(observation)
            output = model(observation) 
            distribution = Categorical(logits = output)
            action = distribution.sample()
            actions.append(action)
            log_prob = dist.log_prob(action) 
            log_probs.append(log_prob)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards = model.append(reward)
            total_reward += reward
            episode_over = terminated or truncated
        lengths.append(t)
    
    #question to consider, how much data from rollout to use when fitting value function? apparently its all of it!
    targets = []
    states = torch.as_tensor(states, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32) #expects tensors
    k= 0
    with torch.no_grad():
        for T in lengths: #big T is length of hte peisode
            for j in range(T): 
                tobootstrap = min(T-j,3) #3 step bootstrapping
                target = sum((gamma ** i) * rewards[k + i] for i in range(tobootstrap))
                if (j + tobootstrap < T):
                    target+= (gamma ** tobootstrap) * V(states[k+tobootstrap])
                targets.append(target)
                k += 1
    
    values = V(states).squeeze(-1) #note that the nn returns [batch_size,1] so need to squeezae to get into a 1d vector
    targets = torch.stack(targets).squeeze(-1)

    loss = loss_fn(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #ok that was the value function fitting, now need to get advantage function

    




    

        
    env.close()