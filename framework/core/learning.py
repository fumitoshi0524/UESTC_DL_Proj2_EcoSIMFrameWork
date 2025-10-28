import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List
from collections import deque


class DQNNetwork(nn.Module):
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1.0 - float(done))
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        device: str = 'cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        self.network = DQNNetwork(obs_dim, action_dim).to(device)
        self.target_network = DQNNetwork(obs_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.training_steps = 0
    
    def select_action(self, observation: np.ndarray, explore_rate: float = 0.1) -> np.ndarray:
        if np.random.rand() < explore_rate:
            return np.random.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            q_values = self.network(obs_tensor)
            action = q_values.argmax(dim=-1).cpu().numpy()
        
        return np.tanh(action.astype(np.float32) / self.action_dim)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        if len(states.shape) == 2 and states.shape[0] == batch_size:
            pass
        else:
            states = states.view(batch_size, -1)
            next_states = next_states.view(batch_size, -1)
        
        current_q_values = self.network(states)
        next_q_values = self.target_network(next_states)
        
        target_q_values = rewards + self.gamma * next_q_values.max(dim=-1)[0] * dones
        
        loss = self.criterion(current_q_values.mean(dim=-1), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        if self.training_steps % 1000 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': float(loss.item()),
            'epsilon': self.epsilon,
        }
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())


class MultiAgentDQNTrainer:
    
    def __init__(self, agent_configs: dict, device: str = 'cpu'):
        self.agents = {
            agent_type: DQNAgent(
                obs_dim=config['obs_dim'],
                action_dim=config['action_dim'],
                learning_rate=config.get('lr', 1e-3),
                gamma=config.get('gamma', 0.99),
                device=device,
            )
            for agent_type, config in agent_configs.items()
        }
        self.device = device
    
    def select_actions(
        self,
        observations: dict,
        explore_rate: float = 0.1,
    ) -> dict:
        
        actions = {}
        for agent_type, obs in observations.items():
            if agent_type in self.agents:
                if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                    actions[agent_type] = np.array([
                        self.agents[agent_type].select_action(o, explore_rate)
                        for o in obs
                    ])
                else:
                    actions[agent_type] = self.agents[agent_type].select_action(obs, explore_rate)
        
        return actions
    
    def store_transitions(self, transitions: List[dict]):
        for transition in transitions:
            agent_type = transition['agent_type']
            if agent_type in self.agents:
                self.agents[agent_type].store_transition(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state'],
                    transition['done'],
                )
    
    def train_all(self, batch_size: int = 32) -> dict:
        training_info = {}
        for agent_type, agent in self.agents.items():
            info = agent.train(batch_size)
            training_info[agent_type] = info
        return training_info
    
    def update_all_target_networks(self):
        for agent in self.agents.values():
            agent.update_target_network()
