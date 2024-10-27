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

class SharedFeatureExtractor(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        # 状態の時間的な関係性を捉えるためのLSTM層
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # 特徴抽出のための層
        self.feature_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.hidden = None
        
    def reset_hidden(self):
        self.hidden = None
        
    def forward(self, x):
        # LSTMは3次元入力を期待するため、時系列次元を追加
        x = x.unsqueeze(1)  # [batch_size, 1, feature_size]
        
        # LSTMの実行
        if self.hidden is None:
            lstm_out, self.hidden = self.lstm(x)
        else:
            # hidden stateをdetachして勾配の伝播を防ぐ
            hidden = (self.hidden[0].detach(), self.hidden[1].detach())
            lstm_out, self.hidden = self.lstm(x, hidden)
        
        # 時系列次元を削除
        x = lstm_out.squeeze(1)
        
        # 特徴抽出
        x = self.feature_net(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature_extractor = SharedFeatureExtractor(state_size)
        
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # より積極的な探索のための初期化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.414)
                layer.bias.data.zero_()
                
    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.policy_net(x)
        
        # より積極的な探索のための温度付きsoftmax
        temperature = 2.0
        probs = F.softmax(logits / temperature, dim=1)
        return probs
    
    def reset_hidden(self):
        self.feature_extractor.reset_hidden()

class ValueNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.feature_extractor = SharedFeatureExtractor(state_size)
        
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value推定のための初期化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.414)
                layer.bias.data.zero_()
                
    def forward(self, x):
        x = self.feature_extractor(x)
        value = self.value_net(x)
        return value
    
    def reset_hidden(self):
        self.feature_extractor.reset_hidden()

class ACTerranAgent(BaseTerranAgent):
    def __init__(self, state_size, action_size):
        super(ACTerranAgent, self).__init__()
        self.gamma = 0.999  # より長期の報酬を重視
        self.lr_pi = 0.0001
        self.lr_v = 0.0002
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.2
        self.gae_lambda = 0.95  # GAE（Generalized Advantage Estimation）のパラメータ

        self.pi = PolicyNet(self.state_size, self.action_size)
        self.v = ValueNet(self.state_size)

        # より安定した最適化のためのAMSGrad
        self.optimizer_pi = torch.optim.AdamW(
            self.pi.parameters(), 
            lr=self.lr_pi, 
            weight_decay=0.001,
            amsgrad=True
        )
        self.optimizer_v = torch.optim.AdamW(
            self.v.parameters(), 
            lr=self.lr_v,
            weight_decay=0.001,
            amsgrad=True
        )

        # より緩やかな学習率の減衰
        self.scheduler_pi = torch.optim.lr_scheduler.StepLR(
            self.optimizer_pi, 
            step_size=5000, 
            gamma=0.95
        )
        self.scheduler_v = torch.optim.lr_scheduler.StepLR(
            self.optimizer_v, 
            step_size=5000, 
            gamma=0.95
        )
        
        self.win_rate_history = []
        self.num_episodes = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.new_game()

    def reset(self):
        super(ACTerranAgent, self).reset()
        self.new_game()
    
    def new_game(self):
        self.state = [0] * self.state_size
        self.action = None
        self.prob = None
        # 新しいゲーム開始時にLSTMの隠れ状態をリセット
        self.pi.reset_hidden()
        self.v.reset_hidden()
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

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
        
        return [len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
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
                len(enemy_marines)]
    
    def update(self, state, action_prob, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # GAEの計算
        with torch.no_grad():
            next_value = self.v(next_state)
            current_value = self.v(state)
            delta = reward + self.gamma * next_value * (1 - done) - current_value
            advantage = delta.item()

        # Value lossの計算（Huber Lossを使用）
        target = reward + self.gamma * next_value * (1 - done)
        value = self.v(state)
        value_loss = F.smooth_l1_loss(value, target.detach())

        # Policy lossの計算（PPOスタイル）
        log_prob = torch.log(action_prob)
        policy_loss = -log_prob * advantage

        # 勾配の更新
        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        
        value_loss.backward()
        policy_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=0.5)
        
        self.optimizer_v.step()
        self.optimizer_pi.step()
        
        # スケジューラーの更新
        self.scheduler_pi.step()
        self.scheduler_v.step()
    
    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.pi.state_dict(),
            'value_state_dict': self.v.state_dict(),
            'optimizer_pi_state_dict': self.optimizer_pi.state_dict(),
            'optimizer_v_state_dict': self.optimizer_v.state_dict(),
        }, filepath)

    def step(self, obs):
        super(ACTerranAgent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
            
        next_state = self.get_state(obs)
        
        if not obs.first():
            reward = (
            obs.reward 
            + (0.000001 if next_state[3] > 0 else 0) 
            + (0.000001 if next_state[5] > 0 else 0) 
            + 0.000001 * self.state[7]
            - 0.000001 * self.state[20]
            + (0.002 if self.state[7] - self.state[20] > 5 and self.action == 5 else 0)
            )
            self.update(self.state, self.prob, reward, next_state, obs.last())
            print(f"Action: {self.action}, Prob: {self.prob:.2f}, Reward: {reward}")
        
        self.state = next_state
        self.action, self.prob = self.get_action(self.state)
        
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
                
        return getattr(self, self.actions[self.action])(obs)

def main(unused_argv):
    torch.autograd.set_detect_anomaly(True)
    agent1 = ACTerranAgent(state_size=21, action_size=6)
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
    plt.xlabel('Episode (x30)')
    plt.ylabel('Win rate per 30 games')
    plt.plot(range(len(agent1.win_rate_history)), agent1.win_rate_history)
    plt.title('AC Terran Agent Win Rate over Time')
    plt.show()

if __name__ == "__main__":
    app.run(main)