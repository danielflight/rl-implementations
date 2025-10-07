# REINFORCE: Monte Carlo Policy Gradient

- **Policy-based method**: Instead of learning a value function and deriving a policy from it, we directly parameterize and optimise the policy $\pi_\theta(a|s)$.  
- **Objective**: Maximize expected return  
  $$
  J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]
  $$  
- **Gradient ascent**: Use the **policy gradient theorem** to update parameters in the direction that increases expected reward.  

## Monte Carlo Aspect
- REINFORCE uses **complete episodes** to compute returns $G_t$.  
- The return acts as an **unbiased sample** of the action-value function $Q^{\pi_\theta}(s_t, a_t)$.  
- This makes updates **unbiased** but introduces **high variance** (compared to TD methods).  

## Update Rule
- At each timestep, increase the probability of actions that led to **higher returns**.  
- The update rule (Policy Gradient Theorem):  
  $$
  \theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t | s_t)
  $$  
- Actions followed by higher returns get reinforced (more likely in the future), while actions followed by poor returns get suppressed.  


## Define the Policy

Parameterise the policy as a function mapping states to action probabilities:

$$
\pi_\theta(a \mid s) = \text{softmax}\big(f_\theta(s)\big)
$$

- **Input**: state vector (for CartPole, 4 dimensions)  
- **Output**: probability of each action (left or right)  


Notes:
- One could implement $f_\theta$ as a **neural network** or a **simple linear model** (if the env is simple enough).  
- Use **PyTorch** or **TensorFlow** to automatically compute gradients for updating $\theta$.  
- The softmax ensures that the outputs form a **valid probability distribution** over actions (could replace softmax with a Gaussian distribution or, naively, a finite-differences approach).


$\newline \newline$
---

## Algorithm  

**For** episode = $1$ to $N$:  
&nbsp;&nbsp;&nbsp;1. Generate an episode:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;States: $s_1, s_2, …, s_{T-1}\newline$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Actions: $a_1, a_2, …, a_{T-1}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rewards: $r_2, r_3, …, r_T$  

&nbsp;&nbsp;&nbsp;2. **For** $t=1$ to $T-1$:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. Compute the **return**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $G_t = \sum_{k=t}^{T} γ^{k−t} * r_{k+1}\newline$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where $γ \in (0, 1)$ is the discount factor and $r_{k+1}$ is the reward signalled to the agent after taking action $a_k$.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. Update the policy parameters:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $θ ← θ + α * G_t * ∇_θ \log(π_θ(a_t | s_t))$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for step-size $\alpha$.

---

## References

- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd Edition.
- David Silver's lectures on Reinforcement Learning.