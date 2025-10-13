# Reinforcement Learning Notes: Monte Carlo and TD(0) Prediction

## 1. Monte Carlo (MC) Prediction

**Goal:** Estimate the state-value function $V^\pi(s)$ for a given policy $\pi$.

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \; \Big| \; S_0 = s \right]
$$

**Key points:**
- Uses **full-episode returns** to update values.
- **First-visit MC:** update V(s) only on the first visit in each episode.
- **High variance**, unbiased.
- No step-size parameter; updates are averages of returns.

**Algorithm (First-Visit MC):**
1. Initialize $V(s) = 0$, returns count = 0 for all states.
2. For each episode:
   - Generate episode $(S_0, A_0, R_1, \dots, S_T)$ following policy $\pi$
   - For each state $s$ first visited at time $t$:
     $$
     G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
     $$
     $$
     V(s) \leftarrow \frac{\text{sum of all first-visit returns}}{\text{number of first visits}}
     $$

**Pros / Cons:**
- Pros: unbiased, simple, converges to true $V^\pi$
- Cons: high variance, must wait until end of episode, slow on sparse rewards

---

## 2. Temporal Difference (TD(0)) Prediction

**Goal:** Estimate $V^\pi(s)$ using **bootstrapping** (update step-by-step).

**TD(0) update rule:**

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
$$

Where:
- $R_{t+1}$ = reward after taking action from $S_t$
- $S_{t+1}$ = next state
- $\gamma$ = discount factor
- $\alpha$ = step-size / learning rate
- $R_{t+1} + \gamma V(S_{t+1})$ is the **TD target**
- $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**

**Key points:**
- Updates **online**, after each step.
- Lower variance than MC (bootstraps from current estimate).
- Slightly biased initially (depends on V estimates).
- Step-size $\alpha$ controls learning speed vs stability.

**Algorithm (TD(0)):**
1. Initialize $V(s) = 0$ for all states.
2. For each episode:
   - Reset environment, observe state $S_t$
   - While not done:
     - Choose action $A_t \sim \pi(S_t)$
     - Observe reward $R_{t+1}$ and next state $S_{t+1}$
     - Update value:
       $$
       V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
       $$
     - Move to next state $S_{t+1}$

**Pros / Cons:**
- Pros: online, lower variance, faster convergence
- Cons: introduces bias (bootstrapping), sensitive to step-size $\alpha$

---

## References

- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd Edition.
- David Silver's lectures on Reinforcement Learning