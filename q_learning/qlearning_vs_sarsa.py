import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

debug = False
if debug:
    rundir,runtag,logger = setup_report('debug')
else:
    rundir,runtag,logger = setup_report('info')

os.makedirs(f"{rundir}/plots", exist_ok=True)



########################################## Hyperparameters ##########################################


env_name = "MountainCar-v0"  # change to "CartPole-v1" or "FrozenLake-v1"
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
DISCRETE_OS_SIZE = [20]  # will adjust per environment

## log hyperparameters
logger.info(f"""

=== Environment settings and hyperparameters ===
    
    Environment: {env_name}
    Discretisation size: {DISCRETE_OS_SIZE[0]}

    Learning rate: {LEARNING_RATE}
    Discount: {DISCOUNT}
    Starting epsilon: {epsilon}

=== Training settings ===

    Number of episodes: {EPISODES}
    Showing every {SHOW_EVERY} episodes

""")

def get_discrete_state(state, env, discrete_os_win_size):
    """Convert continuous state to discrete if needed."""
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return state
    else:
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(int))


########################################## Train agent ##########################################


def train_agent(env_name, algorithm="qlearning", render=False):
    """ Trains an RL agent using either the qlearning or sarsa algorithm, on a gym environment."""

    env = gym.make(env_name)
    os.makedirs(f"{rundir}/qtables/{algorithm}", exist_ok=True)
    logger.info(f"Training an agent using the {algorithm} algorithm...\n")
    
    # check if state space is continuous
    if isinstance(env.observation_space, gym.spaces.Box):
        DISCRETE_OS_SIZE_LOCAL = [20] * len(env.observation_space.high)
        discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE_LOCAL
        q_table_shape = DISCRETE_OS_SIZE_LOCAL + [env.action_space.n]
    else:  # discrete state
        discrete_os_win_size = None
        q_table_shape = [env.observation_space.n, env.action_space.n]
    
    q_table = np.random.uniform(low=-2, high=0, size=q_table_shape)
    epsilon_local = epsilon
    ep_reward = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

    for episode in range(EPISODES):

        if render:
            if episode % SHOW_EVERY == 0:
                env.reset() 
                env = gym.make(env_name, render_mode="human") # will render
            else:
                env.reset()
                env = gym.make(env_name) # will not render


        state_raw, _ = env.reset()
        discrete_state = get_discrete_state(state_raw, env, discrete_os_win_size)
        
        # initial action for SARSA
        if algorithm == "sarsa":
            if np.random.random() > epsilon_local:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
        
        done = False
        episode_reward_local = 0

        while not done:
            # Q-learning action selection
            if algorithm == "qlearning":
                if np.random.random() > epsilon_local:
                    action = np.argmax(q_table[discrete_state])
                else:
                    action = np.random.randint(0, env.action_space.n)
            
            new_state_raw, reward, terminated, truncated, info = env.step(action)
            episode_reward_local += reward
            done = terminated or truncated
            new_discrete_state = get_discrete_state(new_state_raw, env, discrete_os_win_size)
            
            # Update Q-table
            if algorithm == "qlearning" and not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                q_table[discrete_state + (action,)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            elif algorithm == "sarsa":
                # choose next action (on-policy)
                if np.random.random() > epsilon_local:
                    next_action = np.argmax(q_table[new_discrete_state])
                else:
                    next_action = np.random.randint(0, env.action_space.n)
                
                current_q = q_table[discrete_state + (action,)]
                next_q = q_table[new_discrete_state + (next_action,)]
                q_table[discrete_state + (action,)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * next_q)
                
                action = next_action  # move to next action

            discrete_state = new_discrete_state

            # Set Q-value to 0 if goal reached (for MountainCar/FrozenLake)
            if hasattr(env.unwrapped, "goal_position") and new_state_raw[0] >= env.unwrapped.goal_position:
                q_table[discrete_state + (action,)] = 0
                print(f"Reached the goal on episode {episode}")
                break
            elif env_name == "FrozenLake-v1" and reward > 0:  # reached goal
                q_table[discrete_state + (action,)] = 0
                print(f"Reached the goal on episode {episode}")
                break

        # epsilon decay
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon_local -= epsilon_decay_value

        ep_reward.append(episode_reward_local)

        # aggregate metrics
        if not episode % SHOW_EVERY:
            average_reward = sum(ep_reward[-SHOW_EVERY:]) / len(ep_reward[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_reward[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_reward[-SHOW_EVERY:]))
            print(f"[{algorithm}] Episode: {episode}, avg: {average_reward:.2f}, min: {min(ep_reward[-SHOW_EVERY:])}, max: {max(ep_reward[-SHOW_EVERY:])}")

        # save Q-table
        np.save(f"{rundir}/qtables/{algorithm}/{episode}-qtable.npy", q_table)

    env.close()
    # save plot
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label=f'{algorithm} avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label=f'{algorithm} min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label=f'{algorithm} max')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{algorithm.upper()} on {env_name}")
    plt.legend()
    plt.savefig(f"{rundir}/plots/{algorithm}_{env_name}.png")
    plt.clf()

    logger.info("Training Complete.\n")
    return aggr_ep_rewards


if __name__ == '__main__':
    ## render pygame window 
    render = False

    qlearning_results = train_agent(env_name, "qlearning", render=render)
    sarsa_results = train_agent(env_name, "sarsa", render=render)

    # combined comparison plot
    plt.plot(qlearning_results['ep'], qlearning_results['avg'], label='Q-learning avg')
    plt.plot(sarsa_results['ep'], sarsa_results['avg'], label='SARSA avg')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Q-learning vs SARSA on {env_name}")
    plt.legend()
    plt.savefig(f"{rundir}/plots/qlearning_vs_sarsa_{env_name}.png")

    # plot_qtables(rundir=rundir, num_episodes=EPISODES, algorithm="qlearning", create_vid=True)
    

