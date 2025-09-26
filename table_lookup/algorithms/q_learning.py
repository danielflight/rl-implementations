import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


## Using Q-learning (off-policy): It updates the Q-value assuming we always the best action


env_name =  "MountainCar-v0"

env = gym.make(env_name) 
env.reset()

# print(env.observation_space.high) # position and velocity
# print(env.observation_space.low) # position and velocity
# print(env.action_space.n) # 3 actions: left, no push, right

## hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 1000 # how often to render the environment

# discretisation of the continuous state space
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE # the size of each discrete "bucket"

## exploration settings
epsilon = 0.5 # initial exploration rate, higher = more exploration
START_EPSILON_DECAYING = 1 # episode number to start decaying epsilon
END_EPSILON_DECAYING = EPISODES // 2 # episode number to end decaying epsilon
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING) # amount to decay epsilon each episode

# create the Q-table -- initialise to random values between -2 and 0
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # 20x20x3 (3 actions)

ep_reward = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state): 
    """Convert continuous state to discrete state"""
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))





## main training loop
for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        env.reset() 
        env = gym.make(env_name, render_mode="human") # will render
    else:
        env.reset()
        env = gym.make(env_name) # will not render

    # starting state
    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    while not done:
        # choose the action with the highest Q-value for the current discrete state
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # exploit
        else:
            action = np.random.randint(0, env.action_space.n)  # explore
            
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # maximum Q value for the next state
            current_q = q_table[discrete_state + (action,)] # current Q value corresponding to the action taken
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) # Q-learning formula

            # update the Q-table with the new Q value (AFTER the action taken)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.unwrapped.goal_position: # if we reached the goal
            print(f"Reached the goal on episode {episode}")
            q_table[discrete_state + (action,)] = 0 # if we reached the goal, set the Q value to 0
        
        # update the discrete state
        discrete_state= new_discrete_state
    
    # decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_reward.append(episode_reward)

    if not episode % SHOW_EVERY:
        np.save(f"qtables/qlearning/{episode}-qtable.npy", q_table)
        average_reward = sum(ep_reward[-SHOW_EVERY:])/len(ep_reward[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_reward[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_reward[-SHOW_EVERY:]))
        print(f"Episode: {episode}, avg: {average_reward}, min: {min(ep_reward[-SHOW_EVERY:])}, max: {max(ep_reward[-SHOW_EVERY:])}")

env.close() 

## plot the learning curves
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.xlabel("Episode")
plt.title(f"Q-Learning Plot for {env_name}")
plt.legend(loc=4)
plt.savefig(f"plots/qlearning/metrics_{env_name}.png")

