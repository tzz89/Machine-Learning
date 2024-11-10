# Reinforcement Learning in recommender systems
The main goal of reinforcement learning is to maximise the long term satisfaction of the user (reward) and not the immediate reward although the immediate reward is more important (gamma).
Reinforcement learning can be used to combine pointwise recommendations with session based data. 

## Problem Formulation
Recommendation can be converted into a sequential decision-making problem

Documents: item to be recommended
States: item features, user features
Actions: recommended Items
Reward: Long term satisfaction (explicit/implicit)
Gamma: 0(bandits) or 1(RL)
Agent: user or customers receiving recommendation
Env: Platform
Algorithm: RLlib algorithm


### RL Libraries
1. OpenAI Gym: https://github.com/openai/gym
2. Google Dopamine: https://github.com/google/dopamine
3. Gymnasium: https://github.com/Farama-Foundation/Gymnasium



### Reference
1. Reinforcement learning in recommender systems: https://www.youtube.com/watch?v=qJysTu1Xl5U, 
   - git:https://github.com/anyscale/academy/tree/main/ray-rllib/acm_recsys_tutorial_2022
2. huggingface RL course: https://huggingface.co/learn/deep-rl-course/en/unit0/introduction