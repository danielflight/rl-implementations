# RL Implementations 

A collection of reinforcement learning (RL) algorithms implemented as part of a structured 6-week learning plan.  

### Week 1: Core Concepts
- RL basics: agent, environment, state, action, reward  
- Markov Decision Processes (MDPs)  
- Value functions and Bellman equations  
- **Code**: toy example mapping state/action/reward to a trading setup  

### Week 2: Tabular Value-Based Methods
- Policy evaluation & improvement  
- Monte Carlo vs Temporal Difference learning  
- **Code**: MC prediction & TD(0) on FrozenLake  

### Week 3: Q-Learning & SARSA
- Q-learning (off-policy) vs SARSA (on-policy)  
- Exploration–exploitation trade-off  
- **Code**: Q-learning & SARSA on FrozenLake and CartPole  

### Week 4: Policy Gradients
- Policy gradient theorem  
- REINFORCE algorithm  
- **Code**: REINFORCE for CartPole  

### Week 5: Actor–Critic
- Combining value and policy learning  
- Advantage functions  
- **Code**: Simple Actor–Critic implementation  

### Week 6: Deep Reinforcement Learning
- Function approximation & large state spaces  
- Deep Q-Networks (DQN): experience replay, target networks  
- **Code**: Tiny DQN in PyTorch for CartPole  

## Resources

- [David Silver's RL course](https://davidstarsilver.wordpress.com/teaching/)
- [sentdex videos on RL implementations](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7)


## Clone Repo and Install Dependencies

```bash
# clone repo
git clone https://github.com/danielflight/rl-implementations.git
cd rl-implementations

# install dependencies
pip install -r requirements.txt
```

