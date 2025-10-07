## Simple implementation of Monte Carlo policy prediction on the FrozenLake gym environment

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


def env_reset(env, seed=None):
    """Helper function to handle resetting tehe environment in the gym environment"""
    res = env.reset(seed=seed) if seed is not None else env.reset()
    # gym v0.x: returns observation
    # gymnasium / new gym: returns (obs, info)
    if isinstance(res, tuple) and len(res) == 2:
        return res[0]
    return res

def env_step(env, action):
    """Helper function to handle taking steps in the gym environment"""
    res = env.step(action)
    # older gym: (next_state, reward, done, info)
    # newer gym/gymnasium: (next_state, reward, terminated, truncated, info)
    if len(res) == 4:
        next_state, reward, done, info = res
        return next_state, reward, done, info
    else:
        next_state, reward, terminated, truncated, info = res
        done = terminated or truncated
        return next_state, reward, done, info
    

# Simulates one episode following a policy 
def generate_episode(env, policy, max_steps=1000):
    """
    Returns a list of (state, action, reward) tuples for one episode.
        policy: array shape (nS, nA) giving action probabilities per state
    """
    episode = []
    state = env_reset(env) # retart to a random inital position on grid
    done = False
    steps = 0
    nA = policy.shape[1] # num actions

    while not done and steps < max_steps:
        # sample an action according to the policy dist for this state
        ## NOTE: if policy deterministic, there'd only be one non zero entry (=1.0)
        action = np.random.choice(np.arange(nA), p=policy[state])

        # execute the action and record state, action, reward
        next_state, reward, done, _ = env_step(env, action)
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode


def mc_first_visit_prediction(env, policy, num_episodes=5000, gamma=1.0, track_state=0):
    """
    Estimate the state-value function (V^pi(s) for all states s) for a given policy
    using first-visit Monte Carlo (MC) prediction.

    Args:
        policy : np.ndarray, shape (nS, nA)
            The policy to evaluate. policy[s, a] gives the probability of taking
            action a in state s.
        num_episodes : int, optional (default=5000)
            Number of episodes to sample for learning.
        gamma : float, optional (default=1.0)
            Discount factor for future rewards, 0 ≤ gamma ≤ 1.
        max_steps : int, optional (default=1000)
            Maximum steps per episode to prevent infinite loops.
        track_state : int, which state to track.

    Returns:
        V : np.ndarray, shape (nS,)
            Estimated value function array, where V[s] ≈ expected return 
            starting from state s following the given policy.

    Notes:
    - Updates are done using the first-visit Monte Carlo method:
        For each state s visited in an episode, update V[s] by averaging 
        all observed returns G_t following the first visit to s in that episode.
    """
    nS = env.observation_space.n
    V = np.zeros(nS)
    returns_sum = np.zeros(nS)
    returns_count = np.zeros(nS)
    
    # store all first-visit returns for the tracked state
    tracked_returns = []

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(env, policy) # simulate one episode

        # record first-visit index for each state in this episode
        first_visit_idx = {}
        for idx, (s, a, r) in enumerate(episode):
            if s not in first_visit_idx:
                first_visit_idx[s] = idx # first time each state appears in the episode

        # compute returns G_t and update only on first visits
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if first_visit_idx[s] == t:  # first visit
                returns_sum[s] += G
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]
                if s == track_state:
                    tracked_returns.append(G)

        if ep % (num_episodes // 5 if num_episodes >= 5 else 1) == 0:
            print(f"Episode {ep}/{num_episodes}")

    return V, tracked_returns



if __name__ == "__main__":
    # create env (4x4 frozen lake) with the layout:
    #                 S F F F
    #                 F H F H
    #                 F F F H
    #                 H F F G
    # Reward structure:
        # 0 for every step until termination
        # 1 if you reach the goal
        # 0 if you fall into a hole
    # Termination: falling into a hole or reaching the goal
    
    env = gym.make("FrozenLake-v1", is_slippery=True)  # change is_slippery=False for deterministic
    nS = env.observation_space.n
    nA = env.action_space.n

    # policy: uniformly random
    policy = np.ones((nS, nA)) / nA

    V_est, start_state_returns = mc_first_visit_prediction(env, policy, num_episodes=10000, gamma=0.99)


    # print("Estimated V (as 1D array):")
    # print(np.round(V_est, 3))
    print("\nEstimated V (4x4 grid):")
    print(V_est.reshape((4,4)))

    print("\nEstimated V for start state:", V_est[0])
    print("Sample first-visit returns for start state (first 20):", start_state_returns[:20])
    print("Mean:", np.mean(start_state_returns), "\nStd dev:", np.std(start_state_returns))

    plt.figure(figsize=(6,4))
    plt.hist(start_state_returns, bins=np.linspace(0,1,3), edgecolor='k', rwidth=0.7)
    plt.title("Histogram of First-Visit Returns for Start State (MC)")
    plt.xlabel("Return G")
    plt.ylabel("Frequency")
    plt.xticks([0, 1])
    plt.savefig("mc-returns-start-state.png")

