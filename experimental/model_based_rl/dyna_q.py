

f"""
DISCLAIMER:

This implementation of Dyna-Q does not learn well, due to the simplicity of the environment, likely leading to overfitting of the dynamics model.
But, it remains here as a simple illustration of hwo Dyna-Q might look in code 

I plan to improve this in future, by finding a better environment, and/or using a better model (e.g. probabilistic / ensemble model).

"""
 

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *

import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

env = gym.make("CartPole-v1")
env_render = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


# Hyperparams
# ================================================================================

DATA_COLLECTION_EPISODES = 500
TRAINING_EPISODES = 100
SHOW_EVERY = 10
PLANNING_STEPS = 1000

learning_rate_q = 1e-3
learning_rate_model = 5e-3
gamma = 0.99


# For data collection
# ================================================================================

real_buffer = deque(maxlen=10_000)

def collect_data(env, episodes=20):
    """
    Collects random experience from the environment and stores in a buffer.
    """

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            real_buffer.append((state, action, reward, next_state, done))
            state = next_state

collect_data(env, DATA_COLLECTION_EPISODES)
print(f"Collected {len(real_buffer)} samples.\n")



# Dynamics Model
# ================================================================================

class DynamicsModel(nn.Module):
    """
    Feedforward NN to predict next_state and reward given (state, action)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + 1)  # predict next_state (4 dims) + reward (1 dim)
        )

    def forward(self, state, action):
        # One-hot encode discrete actions
        a_onehot = torch.zeros((action.size(0), action_dim))
        a_onehot[torch.arange(action.size(0)), action] = 1.0
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x)
    
model = DynamicsModel(state_dim, action_dim)
optimiser = optim.Adam(model.parameters(), lr=learning_rate_model)
loss_fn = nn.MSELoss()



# Planning with the learned model
# ================================================================================

def model_based_planning(model, q_network, optimizer_q, real_buffer, planning_steps=50):
    """
    Performs planning updates using the learned dynamics model.
    
    Args:
        model: DynamicsModel instance predicting next_state and reward
        q_network: Q-network (state-action value estimator)
        optimizer_q: optimizer for Q-network
        real_buffer: buffer of real transitions to sample starting states from
        planning_steps (int): number of synthetic updates to perform
    """
    for _ in range(planning_steps):
        # Sample a real transition to start from
        state, action, _, _, _ = random.choice(real_buffer)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)
        
        # Predict next_state and reward using the model
        with torch.no_grad():
            pred = model(state_tensor, action_tensor)
            next_state_pred = pred[:, :-1]
            reward_pred = pred[:, -1:]
        
        # Simple Q-learning update (TD target using max over next actions)
        next_q_values = q_network(next_state_pred)
        # td_target = (reward_pred + 0.99 * torch.max(next_q_values)).squeeze()  # gamma=0.99
        next_action_probs = torch.softmax(next_q_values, dim=-1)
        next_v = torch.sum(next_action_probs * next_q_values, dim=-1)
        td_target = (reward_pred + 0.99 * next_v).squeeze()  # gamma=0.99

        
        # Current Q-value for (s,a)
        q_values = q_network(state_tensor)
        q_selected = q_values[0, action]
        
        # Compute loss and update
        loss = nn.MSELoss()(q_selected, td_target)
        optimizer_q.zero_grad()
        loss.backward()
        optimizer_q.step()


# Q-network 
# ================================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)
    
q_network = QNetwork(state_dim, action_dim)
optimiser_q = optim.Adam(q_network.parameters(), lr=learning_rate_q)

# # Perform planning updates
# model_based_planning(model, q_network, optimiser_q, real_buffer, planning_steps=50)
# print("Planning step complete!\n")




# DynaQ training loop
# ================================================================================

def dyna_q_training(env, model, q_network, episodes=100, max_steps=300,
                    real_buffer=None, planning_steps=50, gamma=0.99):
    """
    Trains a Q-network using both real and model-generated experience (Dyna-Q).
    
    Args:
        env: Gym environment
        model: Learned DynamicsModel
        q_network: Q-network
        episodes (int): Number of episodes to run
        max_steps (int): Max steps per episode
        real_buffer: deque storing real transitions
        planning_steps (int): Number of model-generated updates per real step
        gamma (float): Discount factor
    """
    optimiser_q = optim.Adam(q_network.parameters(), lr=1e-3)
    
    if real_buffer is None:
        real_buffer = deque(maxlen=5000)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for t in range(max_steps):
            epsilon = max(0.05, 0.75 - 0.01*(ep/episodes))  # decay epsilon

            # choose action using epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # random action
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    action = torch.argmax(q_values, dim=-1).item() # greedy action

            # step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store real transition
            real_buffer.append((state, action, reward, next_state, done))

            # q-learning update with real experience
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                next_q_values = q_network(next_state_tensor)
                next_action_probs = torch.softmax(next_q_values, dim=-1)
                next_v = torch.sum(next_action_probs * next_q_values, dim=-1)
                td_target = (reward + gamma * next_v * (1 - int(done))).squeeze()
                # td_target = (reward + gamma * torch.max(next_q_values) * (1 - int(done))).squeeze()

            q_values = q_network(state_tensor)
            q_selected = q_values[0, action]
            loss = nn.MSELoss()(q_selected, td_target)

            optimiser_q.zero_grad()
            loss.backward()
            optimiser_q.step()

            # now loop over simulated experience
            model_based_planning(model, q_network, optimiser_q, real_buffer, planning_steps)

            state = next_state
            if done:
                break

        reward_history.append(episode_reward)
        if (ep+1) % SHOW_EVERY == 0:
            avg_reward = np.mean(reward_history[-SHOW_EVERY:])
            print(f"Episode {ep+1}, Avg Reward (last 10 eps): {avg_reward:.2f}")

    return q_network, reward_history


def evaluate_policy(env, q_network, num_episodes=10, render=True, sleep_time=0.05):
    """
    Evaluates the policy without sampling actions to see how good the policy is by returning an avg return over a 
    chosen number of episodes
    """
    total_rewards = []
    for ep_num in range(num_episodes):

        # Test learned policy
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:

            if render:  # render every step
                env.render()
                time.sleep(sleep_time)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values, dim=-1).item() # greedy action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    print(f"Average total reward after training: {np.mean(total_rewards)}")


if __name__ == "__main__":

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
        Model learning rate: {learning_rate_model}
        Q-network learning rate: {learning_rate_q}

    === Training settings ===

        Number of training episodes: {TRAINING_EPISODES}
        Showing every {SHOW_EVERY} episodes
        Number of planning steps per real step: {PLANNING_STEPS}
        Number of data collection episodes: {DATA_COLLECTION_EPISODES}
    """)


    train_model = True
    train_agent = True


    if train_model:

        BATCH_SIZE = 256 
        EPOCHS = 1000 

        for epoch in range(EPOCHS):
            batch = random.sample(real_buffer, BATCH_SIZE)
            state, action, reward, next_state, done = zip(*batch)
            state = torch.tensor(np.array(state), dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32)

            pred = model(state, action)
            pred_next_state = pred[:, :-1]
            pred_reward = pred[:, -1:]

            # update model
            loss = loss_fn(pred_next_state, next_state) + loss_fn(pred_reward, reward)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: model loss = {loss.item():.4f}")
        print()
        torch.save(model.state_dict(), f"{rundir}/dynamics_model.pth")
    
    else:
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load("dynamics_model.pth"))
        model.eval()

    if train_agent:
    
        # Train Dyna-Q
        q_network = QNetwork(state_dim, action_dim)
        q_network, rewards = dyna_q_training(env, model, q_network, episodes=TRAINING_EPISODES, planning_steps=PLANNING_STEPS, gamma=gamma)

        torch.save(q_network.state_dict(), f"{rundir}/q_network.pth")


        ## plot learning curve
        window = SHOW_EVERY
        moving_avg = np.convolve(rewards, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel(f"Moving Average Reward ({SHOW_EVERY} episodes)")
        plt.title("Dyna Q: Learning Curve")
        plt.savefig(f"{rundir}/plots/DynaQ-learning-curve.png")

    else:
        q_network = QNetwork(state_dim, action_dim)
        q_network.load_state_dict(torch.load("q_network.pth"))
        q_network.eval()

    # run without updates / sampling
    evaluate_policy(env_render, q_network, num_episodes=10, render=True, sleep_time=0.05)


