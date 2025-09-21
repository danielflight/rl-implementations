# Reinforcement Learning Notes: Q-Learning vs SARSA

## 1. Q-Learning (Off-Policy TD Control)

**Goal:** Learn the **optimal action-value function** $Q^*(s,a)$ for any policy.

**Update rule:**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \Big]
$$

Where:
- $R_{t+1}$ = reward received after taking action $A_t$
- $S_{t+1}$ = next state
- $\gamma$ = discount factor
- $\alpha$ = learning rate
- $\max_a Q(S_{t+1}, a)$ → **greedy action selection** at next state (off-policy)

**Key points:**
- **Off-policy:** learns about the optimal policy independently of the agent’s current policy (can behave e.g. ε-greedy).
- Converges to $Q^*$ given sufficient exploration and decreasing α.
- Higher variance early, but eventually finds the optimal policy.

**Algorithm:**
1. Initialise $Q(s,a) = 0$ for all states and actions.
2. For each episode:
   - Reset environment, observe $S_t$
   - Choose action $A_t$ using behavior policy (e.g., ε-greedy)
   - While not done:
     - Take action $A_t$, observe reward $R_{t+1}$ and next state $S_{t+1}$
     - Update:
       $$
       Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t) \right]
       $$
     - Choose next action $A_{t+1}$ using policy (for behavior)
     - Set $S_t \leftarrow S_{t+1}, A_t \leftarrow A_{t+1}$

---

## 2. SARSA (On-Policy TD Control)

**Goal:** Learn the **action-value function $Q^\pi(s,a)$** for the current policy $\pi$.

**Update rule:**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big]
$$

Where:
- $A_{t+1}$ = **next action actually taken** under the policy $\pi$
- On-policy → learns the value of the current behavior policy

**Key points:**
- **On-policy:** learns about the policy actually being followed (e.g., ε-greedy).
- Safer in stochastic environments, avoids overly optimistic updates.
- May converge slower than Q-Learning in some settings, but more stable.

**Algorithm:**
1. Initialise $Q(s,a) = 0$ for all states and actions.
2. For each episode:
   - Reset environment, observe $S_t$
   - Choose $A_t \sim \pi(S_t)$ (behavior policy)
   - While not done:
     - Take action $A_t$, observe reward $R_{t+1}$ and next state $S_{t+1}$
     - Choose $A_{t+1} \sim \pi(S_{t+1})$
     - Update:
       $$
       Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t) \right]
       $$
     - Set $S_t \leftarrow S_{t+1}, A_t \leftarrow A_{t+1}$

---

## 3. Comparison: Q-Learning vs SARSA

| Feature | Q-Learning | SARSA |
|---------|------------|-------|
| Policy type | Off-policy (learns optimal policy regardless of behavior) | On-policy (learns about the policy actually followed) |
| TD target | $R + \gamma \max_a Q(S',a)$ | $R + \gamma Q(S',A')$ |
| Exploration impact | More aggressive, may be risky in stochastic environments | Safer, updates depend on actual actions taken |
| Convergence | Converges to $Q^*$ with sufficient exploration | Converges to $Q^\pi$ for the current policy |

---

## References

- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd Edition, Chapters 6-7.
- David Silver's lectures on Reinforcement Learning.
