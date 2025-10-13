from monte_carlo import env_reset, env_step
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def td0_prediction(env, policy, num_episodes=5000, gamma=1.0, alpha=0.1, max_steps=1000, tracked_state=0):
    """
    Estimate the state-value function (V^pi(s) for all states s) for a given policy
    using one-step look ahead temporal difference learning (TD(0)).

    Args: 
        policy : np.ndarray, shape (nS, nA)
            The policy to evaluate. policy[s, a] gives the probability of taking
            action a in state s.
        num_episodes : int, optional (default=5000)
            Number of episodes to sample for learning.
        gamma : float, optional (default=1.0)
            Discount factor for future rewards, 0 ≤ gamma ≤ 1.
        alpha : float, optional (default=0.1)
            Step-size (learning rate) for updating value estimates, 0 < alpha ≤ 1.
        max_steps : int, optional (default=1000)
            Maximum steps per episode to prevent infinite loops.
        track_state : int, which state to track.

    Returns:
        V : np.ndarray, shape (nS,)
            Estimated value function array, where V[s] ≈ expected return 
            starting from state s following the given policy.

    Notes:
    - Updates are done using the TD(0) rule:
        V(S_t) <- V(S_t) + alpha * [R_{t+1} + gamma * V(S_{t+1}) - V(S_t)]
    """
    nS = env.observation_space.n
    V = np.zeros(nS)

    # store all first-visit returns for the tracked state
    td_targets = []

    for ep in range(num_episodes):
        state = env_reset(env)
        done = False
        steps = 0
        nA = policy.shape[1]

        while not done and steps < max_steps:
            action = np.random.choice(np.arange(nA), p=policy[state])
            next_state, reward, done, _ = env_step(env, action)
            
            td_target = reward + gamma * V[next_state]

            if state == tracked_state:
                td_targets.append(td_target)

            # TD(0) update
            V[state] += alpha * (td_target - V[state])

            state = next_state
            steps += 1

        if ep % (num_episodes // 5 if num_episodes >= 5 else 1) == 0:
            print(f"Episode {ep}/{num_episodes}")

    return V, td_targets



if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    nS = env.observation_space.n
    nA = env.action_space.n

    policy = np.ones((nS, nA)) / nA  # random policy
    alpha = 0.3 # define step size (how strongly we update the value toward the TD target), 0.1-0.3 is good

    V_td, td_targets_start = td0_prediction(env, policy, num_episodes=10000, gamma=0.99, alpha=alpha)

    print("\nEstimated V (4x4 grid):")
    print(np.round(V_td.reshape((4,4)),3))

    print("\nEstimated V for start state:", V_td[0])
    print("First 20 TD targets for start state:", td_targets_start[:20])
    print("Mean:", np.mean(td_targets_start), "\nStd dev:", np.std(td_targets_start))

    plt.figure(figsize=(6,4))
    plt.hist(td_targets_start, bins=np.linspace(0,1,3), edgecolor='k', rwidth=0.7)
    plt.title("Histogram of TD targets for start state")
    plt.xlabel("TD target")
    plt.ylabel("Frequency")
    plt.xticks([0, 1])
    plt.savefig("td-target-start-state.png")

