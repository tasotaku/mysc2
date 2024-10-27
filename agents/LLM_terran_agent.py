from pysc2.agents import base_agent
from pysc2.lib import actions, features, units


class LLM_TerranAgent(base_agent.BaseAgent):
  def step(self, obs):
    super(LLM_TerranAgent, self).step(obs)
    
    return actions.FUNCTIONS.no_op()