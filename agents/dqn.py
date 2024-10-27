import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from base_terran_agent import BaseTerranAgent
from random_agent import RandomTerranAgent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from absl import app
from collections import deque
import random



class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DQNTerranAgent(BaseTerranAgent):
    def __init__(self, state_size, action_size):
        super(DQNTerranAgent, self).__init__()
        self.gamma = 0.98
        self.lr = 0.001
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = action_size
        self.state_size = state_size

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.state_size, self.action_size)
        self.qnet_target = QNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.sync_interval = 20
        
        self.win_rate_history = []
        self.num_episodes = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        self.new_game()

    def reset(self):
        super(DQNTerranAgent, self).reset()
        self.new_game()
    
    def new_game(self):
        self.state = [0] * self.state_size
        self.action = None
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        
        queued_marines = (completed_barrackses[0].order_length 
                        if len(completed_barrackses) > 0 else 0)
        
        free_supply = (obs.observation.player.food_cap - 
                    obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        
        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        
        return np.array([
            len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines)/1000,
            queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_idle_scvs),
            len(enemy_supply_depots),
            len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            len(enemy_completed_barrackses),
            len(enemy_marines)
        ], dtype=np.float32)
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
    
    def save_model(self, file_name):
        torch.save(self.qnet.state_dict(), file_name)

    def step(self, obs):
        super(DQNTerranAgent, self).step(obs)
        next_state = self.get_state(obs)
        
        if not obs.first():
            # reward = obs.reward + (0.00001 if next_state[3] > 0 else 0) + (0.00001 if next_state[5] > 0 else 0) + (0.002 if next_state[7] > self.state[7] else 0) + (-0.01 if self.state[7] < 5 and self.action == 5 else 0)
            reward = obs.reward
            self.update(self.state, self.action, reward, next_state, obs.last())
        
        self.state = next_state
        self.action = self.get_action(self.state)
        print(f"Action: {self.action}, Reward: {obs.reward}")
        
        if obs.last():
            self.num_episodes += 1
            if obs.reward == 1:
                self.wins += 1
            elif obs.reward == -1:
                self.losses += 1
            else:
                self.draws += 1
            
            if self.num_episodes == 30:
                win_rate = self.wins / self.num_episodes
                lose_rate = self.losses / self.num_episodes
                draw_rate = self.draws / self.num_episodes
                self.win_rate_history.append(win_rate)
                print(f"Games played: {self.num_episodes}, Win rate: {win_rate:.2f}, Loss rate: {lose_rate:.2f}, Draw rate: {draw_rate:.2f}")
                self.num_episodes = 0
                self.wins = 0
                self.losses = 0
                self.draws = 0
            
            if self.num_episodes % self.sync_interval == 0:
                self.sync_qnet()
            
        return getattr(self, self.actions[self.action])(obs)

def main(unused_argv):
    torch.autograd.set_detect_anomaly(True)
    agent1 = DQNTerranAgent(state_size=21, action_size=6)
    agent2 = RandomTerranAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran), 
                     sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64,
            ),
            step_mul=48,
            disable_fog=True,
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)  
    except KeyboardInterrupt:
        pass
    agent1.save_model("actor_critic.pth")
    plt.figure(figsize=(10, 6))
    plt.xlabel('Episode (x100)')
    plt.ylabel('Win rate per 30 games')
    plt.plot(range(len(agent1.win_rate_history)), agent1.win_rate_history)
    plt.title('AC Terran Agent Win Rate over Time')
    plt.show()

if __name__ == "__main__":
    app.run(main)