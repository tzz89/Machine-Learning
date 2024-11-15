# Reinforcement Learning in recommender systems
The main goal of reinforcement learning is to maximise the long term satisfaction of the user (reward) and not the immediate reward although the immediate reward is more important (gamma).
Reinforcement learning can be used to combine pointwise recommendations with session based data. 

## Key Concepts
1. Markov decision process assumes that the probability of next state is only dependent on previous state, a discretized version is call the bellman equation
2. RL agents do not maximizes the Reward but the Expected reward (Q-Functions/Action values) given the learn policy
3. Deep Q Networks uses neural nets because the (Q / Action values) for each action is too huge as there are infinite number of state, action combinations
4. The are online RL where the labels are not defined at the start of offline RL where there are precollected labels
   


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



### RL Libraries abd simulators
1. OpenAI Gym: https://github.com/openai/gym
2. StableBaseline: https://github.com/DLR-RM/stable-baselines3
3. CleanRL: https://github.com/vwxyzjn/cleanrl
4. Google Dopamine: https://github.com/google/dopamine
5. Gymnasium: https://github.com/Farama-Foundation/Gymnasium
6. recsim: https://github.com/google-research/recsim



### Reference
1. Reinforcement learning in recommender systems: https://www.youtube.com/watch?v=qJysTu1Xl5U, 
   - git:https://github.com/anyscale/academy/tree/main/ray-rllib/acm_recsys_tutorial_2022
2. huggingface RL course: https://huggingface.co/learn/deep-rl-course/en/unit0/introduction
3. Markov decision process: https://www.anyscale.com/blog/reinforcement-learning-with-deep-q-networks

