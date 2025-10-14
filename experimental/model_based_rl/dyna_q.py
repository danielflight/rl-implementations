

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

env = gym.make("MountainCar-v0")
env_render = gym.make("MountainCar-v0", render_mode="human")
state_dim = env.observation_space.shape[0]  # 2: position, velocity
action_dim = env.action_space.n  # 3: left, neutral, right


# Hyperparams
# ================================================================================

DATA_COLLECTION_EPISODES = 100
TRAINING_EPISODES = 200
SHOW_EVERY = 10
PLANNING_STEPS = 100  # More planning helps on Mountain Car

learning_rate_q = 5e-3
learning_rate_model = 5e-2
gamma = 0.99
model_train_interval = 5  # Retrain model every N episodes


# For data collection
# ================================================================================

real_buffer = deque(maxlen=50_000)

def collect_data(env, episodes=20):
    """
    Collects random experience from the environment and stores in a buffer.
    """
    print(f"Collecting {episodes} episodes of random data...")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            real_buffer.append((state, action, reward, next_state, done))
            state = next_state
        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep+1}/{episodes} episodes")

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
            nn.Linear(128, state_dim + 1)  # predict next_state + reward
        )

    def forward(self, state, action):
        # One-hot encode discrete actions
        a_onehot = torch.zeros((action.size(0), action_dim))
        a_onehot[torch.arange(action.size(0)), action] = 1.0
        x = torch.cat([state, a_onehot], dim=-1)
        return self.net(x)
    
model = DynamicsModel(state_dim, action_dim)
optimiser_model = optim.Adam(model.parameters(), lr=learning_rate_model)
loss_fn = nn.MSELoss()


def train_model(model, real_buffer, epochs=50, batch_size=64, val_split=0.2):
    """
    Trains the dynamics model on the real buffer with validation and early stopping.
    """
    # Split data into train/validate
    all_data = list(real_buffer)
    val_size = max(1, int(len(all_data) * val_split))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    
    if len(train_data) < batch_size:
        return 0.0
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    training_data = []
    for epoch in range(epochs):
        # Training step
        batch = random.sample(train_data, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)

        pred = model(state, action)
        pred_next_state = pred[:, :-1]
        pred_reward = pred[:, -1:]

        loss = loss_fn(pred_next_state, next_state) + loss_fn(pred_reward, reward) # training loss
        optimiser_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser_model.step()
        
        # Validation step
        if epoch % 1 == 0 and len(val_data) >= batch_size:
            val_batch = random.sample(val_data, min(batch_size, len(val_data)))
            val_state, val_action, val_reward, val_next_state, val_done = zip(*val_batch)
            val_state = torch.tensor(np.array(val_state), dtype=torch.float32)
            val_action = torch.tensor(val_action, dtype=torch.int64)
            val_reward = torch.tensor(val_reward, dtype=torch.float32).unsqueeze(1)
            val_next_state = torch.tensor(np.array(val_next_state), dtype=torch.float32)
            
            with torch.no_grad():
                val_pred = model(val_state, val_action)
                val_pred_next_state = val_pred[:, :-1]
                val_pred_reward = val_pred[:, -1:]
                val_loss = loss_fn(val_pred_next_state, val_next_state) + loss_fn(val_pred_reward, val_reward) # validation loss
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        training_data.append((epoch+1, loss.item(), val_loss.item()))

    return best_val_loss.item(), training_data


# Planning with the learned model
# ================================================================================

def model_based_planning(model, q_network, optimizer_q, real_buffer, planning_steps=50):
    """
    Performs planning updates using the learned dynamics model.
    Uses standard Q-learning (max over next actions for off-policy updates).
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
        
        # Q-learning update (off-policy: max over next actions)
        with torch.no_grad():
            next_q_values = q_network(next_state_pred)
            max_next_q = torch.max(next_q_values, dim=-1, keepdim=True)[0]
            td_target = (reward_pred + gamma * max_next_q).squeeze()

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




# DynaQ training loop
# ================================================================================

def dyna_q_training(env, model, q_network, episodes=100, max_steps=500,
                    real_buffer=None, planning_steps=50, gamma=0.99,
                    model_train_interval=10):
    """
    Trains a Q-network using both real and model-generated experience (Dyna-Q).
    """
    real_buffer = deque(maxlen=50_000)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for t in range(max_steps):
            epsilon = max(0.05, 0.5 - 0.01*(ep/episodes))  # decay epsilon

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
                max_next_q = torch.max(next_q_values, dim=-1)[0]
                td_target = (reward + gamma * max_next_q * (1 - int(done))).squeeze()

            q_values = q_network(state_tensor)
            q_selected = q_values[0, action]
            loss = nn.MSELoss()(q_selected, td_target)

            optimiser_q.zero_grad()
            loss.backward()
            optimiser_q.step()

            # Perform simulated experience planning
            model_based_planning(model, q_network, optimiser_q, real_buffer, planning_steps)

            state = next_state
            if done:
                break

        reward_history.append(episode_reward)
        
        # Periodically retrain the model on accumulated real data
        if (ep + 1) % model_train_interval == 0 and len(real_buffer) > 256:
            model_loss, _ = train_model(model, real_buffer, epochs=100, batch_size=64)
            print(f"Episode {ep+1}: Retrained model, val_loss = {model_loss:.6f}")
        
        if (ep+1) % SHOW_EVERY == 0:
            avg_reward = np.mean(reward_history[-SHOW_EVERY:])
            print(f"Episode {ep+1}, Avg Reward (last {SHOW_EVERY} eps): {avg_reward:.2f}")

    return q_network, reward_history


def evaluate_policy(env, q_network, num_episodes=10, render=True, sleep_time=0.05):
    """
    Evaluates the policy without sampling actions
    """
    total_rewards = []
    for ep_num in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:

            if render:
                env.render()
                time.sleep(sleep_time)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values, dim=-1).item()
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
        
        Environment: MountainCar-v0
        Discount factor: {gamma}
        Model learning rate: {learning_rate_model}
        Q-network learning rate: {learning_rate_q}

    === Training settings ===

        Number of training episodes: {TRAINING_EPISODES}
        Showing every {SHOW_EVERY} episodes
        Number of planning steps per real step: {PLANNING_STEPS}
        Number of data collection episodes: {DATA_COLLECTION_EPISODES}
        Model retraining interval: every {model_train_interval} episodes
    """)


    train_model_flag = True
    train_agent = True


    if train_model_flag:

        print("Training initial model...")
        model_loss, training_data = train_model(model, real_buffer, epochs=200, batch_size=64)
        print(f"Initial model training complete. Final val_loss: {model_loss:.6f}\n")
        torch.save(model.state_dict(), f"{rundir}/dynamics_model.pth")


        plt.figure()
        epochs_arr, train_losses, val_losses = zip(*training_data)
        plt.plot(epochs_arr, train_losses, label='Train Loss')
        plt.plot(epochs_arr, val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title("Dynamics Model Training")
        plt.legend()
        plt.savefig(f"{rundir}/plots/initial_dynamics_model_training.png")
    
    else:
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load("dynamics_model.pth"))
        model.eval()

    if train_agent:
    
        # Train Dyna-Q
        q_network = QNetwork(state_dim, action_dim)
        q_network, rewards = dyna_q_training(env, model, q_network, episodes=TRAINING_EPISODES, 
                                              planning_steps=PLANNING_STEPS, gamma=gamma,
                                              model_train_interval=model_train_interval)

        torch.save(q_network.state_dict(), f"{rundir}/q_network.pth")


        ## plot learning curve
        window = SHOW_EVERY
        moving_avg = np.convolve(rewards, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel(f"Moving Average Reward ({SHOW_EVERY} episodes)")
        plt.title("Dyna-Q on Mountain Car: Learning Curve")
        plt.savefig(f"{rundir}/plots/DynaQ-MountainCar-learning-curve.png")
        plt.show()

    else:
        q_network = QNetwork(state_dim, action_dim)
        q_network.load_state_dict(torch.load("q_network.pth"))
        q_network.eval()

    # Evaluate
    print("\nEvaluating policy...")
    evaluate_policy(env_render, q_network, num_episodes=5, render=True, sleep_time=0.02)


