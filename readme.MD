Overall Plan for This:

Initialize Gym Environment for CartPole

Create NN

4d vector -> FCN input 4 output 64 -> FCN 64 to 64 -> FCN to output with probabilities of the action which is  1 dimensional
so binary classification basically

loss will be PPO loss

Training Loop - Collect batch of data, ~ 200 loops? idk what is typical
Fit advantage function by approximating V on that data with 3 step bootstrapping

Repeat.


What happened

basically everything but changed trajectories to 20 instead of 200 and added way more epochs