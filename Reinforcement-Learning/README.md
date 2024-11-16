


## Key Concepts
1. Markov decision process assumes that the probability of next state is only dependent on previous state, a discretized version is call the bellman equation
2. RL agents do not maximizes the Reward but the Expected reward (Q-Functions/Action values) given the learn policy
3. Deep Q Networks uses neural nets because the (Q / Action values) for each action is too huge as there are infinite number of state, action combinations
4. The are online RL where the labels are not defined at the start of offline RL where there are precollected labels
5. Different between state and observation is that state is the full information about the environment (eg chess game), observations is the partial information about the state (eg super mario)
6. There are 2 types of tasks, episodic(game, have start and end) and continuous task(stock trading)
7. There are 2 ways to train the policy, 
   1. policy-based method (which action to take)
      1. Deterministic where policy will always return a same action given the same state
      2. stochastic where there is a probability distribuion over the different actions
   2. value-based method(which state is more valuable)
      1. Traditional method - creates Q table
      2. Deep learning method - where the number of states is too huge
   


### RL Libraries abd simulators
1. OpenAI Gym: https://github.com/openai/gym
2. StableBaseline: https://github.com/DLR-RM/stable-baselines3
3. CleanRL: https://github.com/vwxyzjn/cleanrl
4. Google Dopamine: https://github.com/google/dopamine
5. Gymnasium: https://github.com/Farama-Foundation/Gymnasium
6. recsim: https://github.com/google-research/recsim


### Tutorial
1. Foundation of RL 6 part lecture:https://www.youtube.com/watch?v=2GwBez0D20A&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0 


### Reference
1. Reinforcement learning in recommender systems: https://www.youtube.com/watch?v=qJysTu1Xl5U, 
   - git:https://github.com/anyscale/academy/tree/main/ray-rllib/acm_recsys_tutorial_2022
2. huggingface RL course: https://huggingface.co/learn/deep-rl-course/en/unit0/introduction
3. Markov decision process: https://www.anyscale.com/blog/reinforcement-learning-with-deep-q-networks

