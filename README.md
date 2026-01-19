# Simverse

Simverse is an RL library which contain pre-built ENVS, policies, and recipes. It is designed to learn and get started with experimenting RL.




# Abstraction Classes
AgentSpec -- creates the agent
ENVSpec -- creates the Env
PolicySpec -- creates a policy



# Agent
1. Agent plays/takes action in the simulation
2. Each agent has their own policy
3. Agent can takes action given action space

# ENV
1. A universe where the simulation runs 
2. ENV contains multiple trainable agents and NPCs
3. ENV has their own rules, where agent can learn follow rules to live and win.

# Policy
1. policy is an core brain of a agent.
2. An trainable agent can hold a policy, NPCs have no policy.
3. policy is the core neural network of a agent which help them to learn the ENV.
4. policy can be any neural network like Transformers, LLM, CNN, RNN, LSTM, World Models, etc.

