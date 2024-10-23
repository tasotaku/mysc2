from base_terran_agent import BaseTerranAgent
import random

class RandomTerranAgent(BaseTerranAgent):
  def step(self, obs):
    super(RandomTerranAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)