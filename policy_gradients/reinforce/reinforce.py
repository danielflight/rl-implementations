import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Can improve by:
## - using batch updates over multiple episodes instead of incremental updates


# Set up the environment
env = gym.make("CartPole-v1")
render_env = gym.make("CartPole-v1", render_mode="human") 

state_dim = env.observation_space.shape[0]  # 4 continuous variables: Cart position, velocity, pole angle, angular velocity
action_dim = env.action_space.n  # 2 discrete actions (left / right)

EPISODES = 10000
SHOW_EVERY = 1000
alpha = 1e-3
gamma = 0.99
normalise_returns = True # whether to normalise returns
reward_history = []

# Define policy network
# Input: state vector (4-dim)
# Output: action probabilities via softmax (i.e., a probability distribution over actions)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), # fully connected input layer
            nn.ReLU(), # nonlinearity
            nn.Linear(128, action_dim), # map hidden rep -> action logits (number of possible actions)
            nn.Softmax(dim=-1) # convert to probabilities
        )
    
    def forward(self, x):
        """Takes a batch of states, pushes them through the the layers and returns a vector of probs [p_left, p_right]"""
        return self.fc(x)

policy_nn = PolicyNetwork(state_dim, action_dim)
optimiser_nn = optim.Adam(policy_nn.parameters(), lr=alpha) # gradient-based optimizer (Adam). 
                                                           # It updates the weights of the network (theta) 
                                                           # using gradients from the REINFORCE loss


############################################################################

# Alternatively: linear function approximator
# Simple linear mapping from state -> action probabilities
class LinearPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)  # no hidden layers
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)  # softmax for action probabilities

policy_linear = LinearPolicy(state_dim, action_dim)
optimiser_linear = optim.Adam(policy_linear.parameters(), lr=alpha)

############################################################################


def generate_episode(env, policy_nn, render = False):
    """
    Generates the episode, and allows the policy to interact with the env

    Returns:
        list of tuples (state, action, reward) at each time step
    """
    state, info = env.reset() # reset env

    done = False
    episode = []

    while not done:
        
        if render:
            env.render()

        # sample action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_nn(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        # step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # store episode results
        episode.append((state, action, reward))

        # move to next state
        state = next_state

    return episode


def compute_returns(episode, gamma = 1.0, normalise = True):
    """
    Computes returns by summing over all remaining timesteps from the current 
    using backwards recursion G_t = r_{t+1} + gamma*G_{t+1} instead explicit sum 
    G_t = sum_{k=t}^{T} gamma^(k-t) * r_{k+1}

    Args:
        normalise: if True, will normalise the returns to reduce variance of gradient estimate
    
    Returns:
        the list of returns at each time step
    """
    returns = []
    G = 0
    # go backwards through the episode
    for _, _, reward in reversed(episode):
        G = reward + gamma * G
        returns.insert(0, G)  # prepend so order matches states

    if normalise:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    return returns


def update_policy(episode, returns, policy_nn, optimizer_nn):
    """
    Updates the current policy parameters using the policy gradient theorem:
    
    theta <- theta + alpha * G_t * grad(log(pi))
    """
    losses = []
    for (state, action, _), G_t in zip(episode, returns):
        # compute log probability 
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_nn(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))

        # policy gradient loss
        loss = -log_prob * G_t # alpha already defined in optimiser_nn
        losses.append(loss)

    # sum over episode
    loss = torch.stack(losses).sum()

    # Backprop + update
    optimizer_nn.zero_grad()
    loss.backward()
    optimizer_nn.step()

    return loss.item()


def evaluate_policy(env, policy_nn, num_episodes=5, render=False, sleep_time=0.05):
    """
    Evaluates the policy without sampling actions to see how good the policy is by returning an avg return over a 
    chosen number of episodes
    """
    total_rewards = []
    for ep_num in range(num_episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if render and ep_num == 0: 
                env.render()
                time.sleep(sleep_time)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_nn(state_tensor)
            action = torch.argmax(action_probs).item()  # greedy action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
    avg_reward = sum(total_rewards)/num_episodes
    print(f"Average reward over {num_episodes} eval episodes: {avg_reward}")
    return avg_reward


if __name__ == '__main__':

    debug = False
    if debug:
        rundir,runtag,logger = setup_report('debug')
    else:
        rundir,runtag,logger = setup_report('info')

    os.makedirs(f"{rundir}/plots", exist_ok=True)

    ## log hyperparameters
    logger.info(f"""

    === Environment settings and hyperparameters ===
        
        Environment: CartPole
        Step-size: {gamma}
        Learning rate: {alpha}

    === Training settings ===

        Number of episodes: {EPISODES}
        Showing every {SHOW_EVERY} episodes
        Normalising returns?: {normalise_returns}
    """)


    ## Training loop
    for episode_num in range(EPISODES):
        # # generate an episode
        episode = generate_episode(env, policy_nn)

        # compute returns
        returns = compute_returns(episode, gamma, normalise=normalise_returns)
        
        # update policy
        loss = update_policy(episode, returns, policy_nn, optimiser_nn)
        
        # track total reward for episode
        total_reward = sum([r for (_, _, r) in episode])
        reward_history.append(total_reward)
        
        # print progress 
        if (episode_num+1) % SHOW_EVERY == 0:
            avg_reward = sum(reward_history[-SHOW_EVERY:])/SHOW_EVERY
            print(f"Episode {episode_num+1}, Avg Reward: {avg_reward:.2f}")

# save policy parameters
torch.save(policy_nn.state_dict(), f"{rundir}/policy_params.pth")

## plot learning curve
window = SHOW_EVERY
moving_avg = np.convolve(reward_history, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid')
plt.plot(moving_avg)
plt.xlabel("Episode")
plt.ylabel(f"Moving Average Reward ({SHOW_EVERY} episodes)")
plt.title("REINFORCE: Learning Curve")
plt.savefig(f"{rundir}/plots/REINFORCE-learning-curve.png")

# render final graphic
evaluate_policy(render_env, policy_nn, num_episodes=10, render=True, sleep_time=0.05)

### NOTE: To load in a saved policy, just do
# policy_nn = PolicyNetwork(state_dim, action_dim)
# policy_nn.load_state_dict(torch.load("policy_cartpole.pth"))
# policy_nn.eval()
###