import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from .TD_actor_critic import select_action, evaluate_policy

# Lunar Lander env
env = gym.make("LunarLander-v3", render_mode=None)  
render_env = gym.make("LunarLander-v3", render_mode="human") 
state_dim = env.observation_space.shape[0] # 8-dimensional state (x/y pos, vel, angle, ang vel, R/L leg contact)
action_dim = env.action_space.n # 4 discrete actions ()

# Hyperparameters
learning_rate_actor = 1e-4  
learning_rate_critic = 1e-3
gamma = 0.99  
EPISODES = 2000
SHOW_EVERY = 250
MAX_STEPS_PER_EPISODE = 200
reward_history = []


# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1) # prob. dist. over actions

# Critic network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)   
        return q_values # Q vals for each action 
 

# Initialise Actor and Critic networks
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

# optimisers
actor_optimiser = optim.Adam(actor.parameters(), lr=learning_rate_actor)
critic_optimiser = optim.Adam(critic.parameters(), lr=learning_rate_critic)

# Loss function for the critic
critic_loss_fn = nn.MSELoss()



if __name__ == "__main__":

    debug = False
    if debug:
        rundir,runtag,logger = setup_report('debug')
    else:
        rundir,runtag,logger = setup_report('info')

    os.makedirs(f"{rundir}/plots", exist_ok=True)
    os.makedirs(f"{rundir}/actor_params", exist_ok=True)

    ## log hyperparameters
    logger.info(f"""

    === Environment settings and hyperparameters ===
        
        Environment: LunarLander-v3
        Discount factor: {gamma}
        Learning rate (Actor): {learning_rate_actor}
        Learning rate (Critic): {learning_rate_critic}

    === Training settings ===

        Number of episodes: {EPISODES}
        Maximum number of steps per episode: {MAX_STEPS_PER_EPISODE}
        Showing every {SHOW_EVERY} episodes
    """)

    train = True

    if train:
        ## Training loop
        for episode in range(EPISODES):
            state, _ = env.reset()  
            episode_reward = 0
            done = False

            for step in range(MAX_STEPS_PER_EPISODE):        

                # Select action
                action, log_prob = select_action(state)

                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

                # ===== UPDATE CRITIC =====
                # Compute TD target
                with torch.no_grad():
                    next_q_values = critic(next_state_tensor)
                    next_action_probs = actor(next_state_tensor)
                    # Expected Q-value for next state (V(s'))
                    next_v = torch.sum(next_action_probs * next_q_values, dim=-1)
                    td_target = reward + gamma * next_v * (1 - int(done))

                # Compute current Q-value
                q_values = critic(state_tensor)
                q_selected = q_values[0, action].unsqueeze(0)  
                
                # Update critic
                critic_loss = critic_loss_fn(q_selected, td_target)
                critic_optimiser.zero_grad()
                critic_loss.backward()
                critic_optimiser.step()
                
                # ===== UPDATE ACTOR =====
                # Compute advantage: A(s,a) = Q(s,a) - V(s)
                with torch.no_grad():
                    q_values = critic(state_tensor)
                    action_probs = actor(state_tensor)
                    v_state = torch.sum(action_probs * q_values, dim=-1) # expected values over actions V(s') = E(Q(s',a')) = sum(pi(a|s)Q(s,a))
                    q_selected = q_values[0, action]
                    advantage = q_selected - v_state

                # Actor loss (policy gradient with advantage)
                actor_loss = -log_prob * advantage
                actor_optimiser.zero_grad()
                actor_loss.backward()
                actor_optimiser.step()

                # Move to next state
                state = next_state

                if done:
                    break

            reward_history.append(episode_reward)

            # Print progress 
            if (episode+1) % SHOW_EVERY == 0:
                avg_reward = sum(reward_history[-SHOW_EVERY:])/SHOW_EVERY
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
                torch.save(actor.state_dict(), f"{rundir}/actor_params/params_ep-{episode+1}.pth")

        ## Plot learning curve
        window = SHOW_EVERY
        moving_avg = np.convolve(reward_history, np.ones(SHOW_EVERY)/SHOW_EVERY, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel(f"Moving Average Reward ({SHOW_EVERY} episodes)")
        plt.title("Q Actor-Critic: Learning Curve")
        plt.savefig(f"{rundir}/plots/QAC-learning-curve.png")

    else:
        # Load a saved policy 
        actor = Actor(state_dim, action_dim)
        actor.load_state_dict(torch.load("actor_policy_params.pth"))
        actor.eval()

    # Render final graphic
    evaluate_policy(render_env, actor, num_episodes=5, render=True, sleep_time=0.05)