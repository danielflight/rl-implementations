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


# Lunar Lander env
env = gym.make("LunarLander-v3", render_mode=None)  
render_env = gym.make("LunarLander-v3", render_mode="human") 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameters
learning_rate_actor = 3e-4
learning_rate_critic = 1e-3
gamma = 0.99

entropy_coef_start = 0.05  
entropy_coef_end = 0.001   
entropy_decay_episodes = 2500

EPISODES = 5000
SHOW_EVERY = 500
MAX_STEPS_PER_EPISODE = 500
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


# Critic network (V-function) 
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # output single V(s) value

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        v_value = self.fc3(x)  
        return v_value


# Initialise Actor and Critic networks
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)  # Only takes state_dim, not action_dim

# optimisers
actor_optimiser = optim.Adam(actor.parameters(), lr=learning_rate_actor)
critic_optimiser = optim.Adam(critic.parameters(), lr=learning_rate_critic)

# Loss function for the critic
critic_loss_fn = nn.MSELoss()


def select_action(state):
    """Select an action using the actor's policy."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = actor(state_tensor)
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def evaluate_policy(env, actor, num_episodes=5, render=False, sleep_time=0.05):
    """Evaluates the policy without sampling."""
    total_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_probs = actor(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward            

            if render:
                env.render()

            state = next_state
        
        total_rewards.append(episode_reward)
    avg_reward = sum(total_rewards)/num_episodes
    print(f"Average reward over {num_episodes} eval episodes: {avg_reward}")
    return avg_reward



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
        Initial entropy coefficient: {entropy_coef_start}

    === Training settings ===

        Number of episodes: {EPISODES}
        Maximum number of steps per episode: {MAX_STEPS_PER_EPISODE}
        Showing every {SHOW_EVERY} episodes
    """)

    train = False

    if train:
        ## Training loop
        for episode in range(EPISODES):
            # decay entropy (linearly)
            progress = min(episode / entropy_decay_episodes, 1.0)
            entropy_coef = entropy_coef_start + progress * (entropy_coef_end - entropy_coef_start)

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
                # Compute TD target: r + gamma * V(s')
                with torch.no_grad():
                    next_v = critic(next_state_tensor)
                    td_target = reward + gamma * next_v * (1 - int(done))

                # Compute current V-value
                v_current = critic(state_tensor)
                
                # Update critic
                critic_loss = critic_loss_fn(v_current, td_target)
                critic_optimiser.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5) # Gradient clipping for stability
                critic_optimiser.step()

                
                # ===== UPDATE ACTOR =====
                # Compute advantage using TD error: delta = r + gamma*V(s') - V(s)
                with torch.no_grad():
                    v_current = critic(state_tensor)
                    next_v = critic(next_state_tensor)
                    td_error = reward + gamma * next_v * (1 - int(done)) - v_current
                    advantage = td_error

                # Compute entropy for exploration
                action_probs = actor(state_tensor)
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

                # Actor loss (policy gradient with TD error as advantage)
                actor_loss = -log_prob * advantage #- entropy_coef * entropy
                actor_optimiser.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
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
        plt.title("TD Actor-Critic: Learning Curve")
        plt.savefig(f"{rundir}/plots/TD-AC-learning-curve.png")

    else:
        # Load a saved policy 
        actor = Actor(state_dim, action_dim)
        actor.load_state_dict(torch.load("TDAC_output_1/actor_params/params_ep-5000.pth"))
        actor.eval()

    # Render final graphic
    evaluate_policy(render_env, actor, num_episodes=5, render=True, sleep_time=0.05)