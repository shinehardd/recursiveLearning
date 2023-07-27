# Single Agent
from rl.agents.reinforce.reinforce import REINFORCE

# from rl.agents.a2c.a2c import A2C
# from rl.agents.ppo.ppo import PPO
# from rl.agents.dqn.dqn import DQN
# from rl.agents.ddqn.ddqn import DDQN

REGISTRY = {}

REGISTRY["reinforce"] = REINFORCE
# REGISTRY["a2c"] = A2C
# REGISTRY["ppo"] = PPO
# REGISTRY["dqn"] = DQN
# REGISTRY["ddqn"] = DDQN