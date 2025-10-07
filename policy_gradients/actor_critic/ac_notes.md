
  
- CONSIDER Deterministic Policy Gradients?

# Actor–Critic Methods

Actor–Critic methods combine the **policy-based** and **value-based** approaches in reinforcement learning and follow approximate policy gradients:

- **Actor**: updates the policy parameters in the direction suggested by the critic.
- **Critic**: estimates a value function (e.g., state-value $V(s)$ or action-value $Q(s,a)$) to guide the actor.

The update rule looks like:

$$
\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a|s) \, \hat{A}(s,a)
$$

where:
- $\theta$: policy (actor) parameters  
- $\pi_\theta(a|s)$: probability of action $a$ in state $s$  
- $\hat{A}(s,a)$: advantage estimate (critic’s feedback)

---

## 1. Q Actor–Critic

- **Critic**: learns the action-value function $Q(s,a)$.  
- **Advantage Estimate**:  

$$
\hat{A}(s,a) = Q(s,a)
$$

- Update:  

$$
\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a|s) \, Q(s,a)
$$

**Pros**: directly uses Q-values.  
**Cons**: Q-learning introduces high variance.

---

## 2. Advantage Actor–Critic (A2C / Q–V)

- **Critic**: learns both $Q(s,a)$ and $V(s)$, or directly learns the advantage.  
- **Advantage Estimate**:  

$$
\hat{A}(s,a) = Q(s,a) - V(s)
$$

- Update:  

$$
\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a|s) \, (Q(s,a) - V(s))
$$

**Pros**: lower variance due to value baseline subtraction.  
**Cons**: requires maintaining $Q$ and $V$.

**NOTE**: One might calculate $V(s)$ via a policy weighted baseline $V(s) = \sum_{a \in A} \pi(a|s)Q(s,a)$ or a have separate network for $V$ and learn it. 

---

## 3. TD Actor–Critic

- **Critic**: learns the value function $V(s)$ via TD learning.  
- **Advantage Estimate**: 1-step TD error  

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- Update:  

$$
\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a_t|s_t) \, \delta_t
$$

**Pros**: sample-efficient, no need for Q-values.  
**Cons**: biased due to bootstrapping.

---

## 4. TD($\lambda$) Actor–Critic

- **Critic**: uses eligibility traces to blend multi-step returns.  
- **Advantage Estimate**:  

$$
\delta_t^\lambda = \sum_{n=1}^{\infty} (\lambda^{n-1}) \, \delta_{t}^{(n)}
$$

where $\delta_{t}^{(n)}$ is the n-step TD error.  

- Update:  

$$
\theta \leftarrow \theta + \alpha \, \nabla_\theta \log \pi_\theta(a_t|s_t) \, \delta_t^\lambda
$$

- **Eligibility Traces (for actor & critic)**:  

$$
e_t = \gamma \lambda e_{t-1} + \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

- Update actor using:  

$$
\theta \leftarrow \theta + \alpha \, \delta_t \, e_t
$$

**Pros**: balances bias–variance tradeoff, better credit assignment.  
**Cons**: more hyperparameters ($\lambda$).

---

## Summary

| Method                 | Critic Learns   | Advantage Estimate     | Variance | Bias |
|-------------------------|-----------------|------------------------|----------|------|
| Q Actor–Critic         | $Q(s,a)$      | $Q(s,a)$             | High     | Low  |
| Advantage Actor–Critic | $Q(s,a), V(s)$| $Q(s,a) - V(s)$      | Lower    | Low  |
| TD Actor–Critic        | $V(s)$        | TD error $\delta_t$  | Low      | Bias from bootstrap |
| TD($\lambda$) A–C    | $V(s)$        | TD($\lambda$) error  | Tunable  | Tunable |
