# Reinforcement Learning in recommender systems
The main goal of reinforcement learning is to maximise the long term satisfaction of the user (reward) and not the immediate reward although the immediate reward is more important (gamma).
Reinforcement learning can be used to combine pointwise recommendations with session based data. 


   


## Problem Formulation
Recommendation can be converted into a sequential decision-making problem

Documents: item to be recommended
States: item features, user features
Actions: recommended Items
Reward: Long term satisfaction (explicit/implicit)
Gamma(Discount factor): 0(bandits) or 1(RL)
Agent: user or customers receiving recommendation
Env: Platform
Algorithm: RLlib algorithm



