import gymnasium as gym 
import numpy as np
from neuralnet import PolicyNet
from neuralnet import ValueNet
import torch
import torch.nn as nn
from torch.distributions import Categorical
env = gym.make("CartPole-v1")
# HYPERPARAMETERS
epochs = 100 #how many policy upgrades should we do
N = 200 #how many trajectories should we collect for each V fitting
gamma = 0.99 #discount factor
lr = 1e-3 #learning rate
eps = 0.2 #clipping range
M = 5 #grad steps for ppo policy 
model = PolicyNet()
V = ValueNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(V.parameters(), lr = lr)
policy_optimizer = torch.optim.Adam(model.parameters(),lr =lr)



for i in range(epochs):
    with torch.no_grad():
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
                log_prob = distribution.log_prob(action) 
                log_probs.append(log_prob)
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                total_reward += reward
                episode_over = terminated or truncated
            lengths.append(t)

    #question to consider, how much data from rollout to use when fitting value function? apparently its all of it!
    targets = []
    observations = torch.as_tensor(observations, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32) #expects tensors

    k= 0
    with torch.no_grad():
        for T in lengths: #big T is length of hte peisode
            for j in range(T): 
                tobootstrap = min(T-j,3) #3 step bootstrapping
                target = sum((gamma ** n) * rewards[k + n] for n in range(tobootstrap))
                if (j + tobootstrap < T):
                    target+= (gamma ** tobootstrap) * V(observations[k+tobootstrap])
                targets.append(target)
                k += 1

    values = V(observations).squeeze(-1) #note that the nn returns [batch_size,1] so need to squeezae to get into a 1d vector
    targets = torch.stack(targets).squeeze(-1)

    loss = loss_fn(values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #ok that was the value function fitting, now need to get advantage function
    #going to skip GAE for now, UPDATE later perhaps

    advantage = [] #list of advantage for state action pairs 
    k = 0
    for T in lengths:
        for j in range(T):
            quantity = 0
            if (j == T-1):
                quantity = rewards[k] - values[k]
            else:
                quantity = rewards[k] + gamma * values[k+1] - values[k]
            advantage.append(quantity)
            k += 1
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    advantage = torch.stack(advantage)
    for j in range(M): 
        k=0
        updated_policy_probs = model(observations)
        newdistribution = Categorical(logits = updated_policy_probs)
        new_log_probs = newdistribution.log_prob(actions)
        ratios = torch.exp(new_log_probs - log_probs)
        tot = 0
        for T in lengths:
            for j in range(T):
                clipped = torch.clamp(ratios[k],1-eps,1+eps)
                qt = torch.min(ratios[k] * advantage[k], clipped * advantage[k])
                tot += qt
                k += 1
        tot = -tot
        policy_optimizer.zero_grad()

        tot.backward()
        policy_optimizer.step()
    
env.close()

torch.save(model.state_dict(), "a.pth")  
