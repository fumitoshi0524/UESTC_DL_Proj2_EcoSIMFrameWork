import numpy as np
from typing import Dict, Callable, Any, Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AgentSpec:
    name: str
    num_agents: int
    obs_space: Tuple[int]
    action_space: Tuple[int, int]
    reward_fn: Callable
    action_processor: Optional[Callable] = None


@dataclass
class MarketSpec:
    clearing_fn: Callable
    state_updater: Optional[Callable] = None


class Agent:
    
    def __init__(self, agent_id: int, spec: AgentSpec):
        self.agent_id = agent_id
        self.spec = spec
        self.state = np.random.rand(*spec.obs_space).astype(np.float32)
        self.last_reward = 0.0
        self.action_history = []
        self.reward_history = []
    
    def get_observation(self) -> np.ndarray:
        return self.state.copy()
    
    def update_state(self, new_state: np.ndarray):
        self.state = np.clip(new_state, 0, 1).astype(np.float32)
    
    def record_reward(self, reward: float):
        self.last_reward = reward
        self.reward_history.append(reward)
    
    def record_action(self, action: np.ndarray):
        self.action_history.append(action.copy())


class AgentGroup:
    
    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.agents = [Agent(i, spec) for i in range(spec.num_agents)]
        self.collective_state = np.random.rand(spec.num_agents, *spec.obs_space).astype(np.float32)
        self.collective_reward = np.zeros(spec.num_agents)
    
    def get_observations(self) -> np.ndarray:
        return self.collective_state.copy()
    
    def update_states(self, new_states: np.ndarray):
        self.collective_state = np.clip(new_states, 0, 1).astype(np.float32)
        for i, agent in enumerate(self.agents):
            agent.update_state(new_states[i])
    
    def get_rewards(self) -> np.ndarray:
        return self.collective_reward.copy()
    
    def set_rewards(self, rewards: np.ndarray):
        self.collective_reward = np.clip(rewards, -1e6, 1e6)
        for i, agent in enumerate(self.agents):
            agent.record_reward(float(self.collective_reward[i]))


class GameEnvironment:
    
    def __init__(
        self,
        agents_specs: Dict[str, AgentSpec],
        market_spec: MarketSpec,
        time_steps: int = 100,
        state_normalizer: Optional[Callable] = None,
    ):
        self.agents_specs = agents_specs
        self.market_spec = market_spec
        self.time_steps = time_steps
        self.normalize = state_normalizer or (lambda x: np.clip(x, 0, 1))
        
        self.agent_groups = {
            name: AgentGroup(spec)
            for name, spec in agents_specs.items()
        }
        
        self.current_step = 0
        self.market_state = {}
        self.policy_state = {}
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'market_states': [],
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.current_step = 0
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'market_states': [],
        }
        
        for group in self.agent_groups.values():
            group.collective_state = np.random.rand(
                group.spec.num_agents, *group.spec.obs_space
            ).astype(np.float32) * 0.5 + 0.25
        
        self.market_state = {}
        
        return {name: group.get_observations() for name, group in self.agent_groups.items()}
    
    def step(
        self,
        actions: Dict[str, np.ndarray],
        policy_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], bool, Dict]:
        
        self.current_step += 1
        
        processed_actions = {}
        for agent_type, action in actions.items():
            processor = self.agents_specs[agent_type].action_processor
            if processor:
                processed_actions[agent_type] = processor(action)
            else:
                processed_actions[agent_type] = action
        
        self.market_state = self.market_spec.clearing_fn(processed_actions, policy_params or {})
        
        if self.market_spec.state_updater:
            self.market_state = self.market_spec.state_updater(self.market_state, processed_actions)
        
        rewards = self._compute_rewards(processed_actions, policy_params or {})
        
        new_states = self._update_agent_states(processed_actions, self.market_state)
        
        for agent_type, group in self.agent_groups.items():
            group.update_states(new_states[agent_type])
            group.set_rewards(rewards[agent_type])
        
        done = self.current_step >= self.time_steps
        
        obs = {name: group.get_observations() for name, group in self.agent_groups.items()}
        
        self.episode_data['observations'].append(obs)
        self.episode_data['actions'].append(processed_actions)
        self.episode_data['rewards'].append(rewards)
        self.episode_data['market_states'].append(self.market_state.copy())
        
        info = {
            'step': self.current_step,
            'market': self.market_state.copy(),
            'done': done,
        }
        
        return obs, rewards, done, info
    
    def _compute_rewards(self, actions: Dict[str, np.ndarray], policy: Dict) -> Dict[str, np.ndarray]:
        rewards = {}
        for agent_type, spec in self.agents_specs.items():
            agent_actions = actions.get(agent_type, np.zeros((spec.num_agents, 1)))
            rewards[agent_type] = spec.reward_fn(
                agent_actions,
                self.market_state,
                policy,
            )
        return rewards
    
    def _update_agent_states(
        self,
        actions: Dict[str, np.ndarray],
        market: Dict,
    ) -> Dict[str, np.ndarray]:
        
        new_states = {}
        for agent_type, group in self.agent_groups.items():
            current_state = group.collective_state
            action = actions.get(agent_type, np.zeros((group.spec.num_agents, 1)))
            
            action_effect = action / (np.max(np.abs(action)) + 1e-6)
            
            state_transition = 0.7 * current_state
            if action_effect.shape == current_state.shape:
                state_transition += 0.2 * action_effect
            else:
                state_transition += 0.2 * np.tile(action_effect, (1, current_state.shape[1] // action_effect.shape[1] + 1))[:, :current_state.shape[1]]
            
            state_transition += 0.1 * np.random.rand(*current_state.shape)
            new_states[agent_type] = self.normalize(state_transition)
        
        return new_states
    
    def get_episode_data(self) -> Dict:
        return self.episode_data.copy()


class ExperimentRunner:
    
    def __init__(
        self,
        env_factory: Callable[[], GameEnvironment],
        num_episodes: int,
        max_steps: int,
    ):
        self.env_factory = env_factory
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.all_episodes = []
    
    def run_episode(
        self,
        policy_fn: Callable[[Dict[str, np.ndarray], int, float], Dict[str, np.ndarray]],
        policy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        env = self.env_factory()
        obs = env.reset()
        
        for step in range(self.max_steps):
            explore_rate = max(0.1, 1.0 - (step + 1) / self.max_steps)
            actions = policy_fn(obs, step, explore_rate)
            obs, rewards, done, info = env.step(actions, policy_params)
            
            if done:
                break
        
        return env.get_episode_data()
    
    def run_experiment_series(
        self,
        policy_configs: List[Dict[str, Any]],
        action_policy_fn: Callable,
    ) -> Dict[str, List[Dict]]:
        
        results = {}
        
        for config_idx, config in enumerate(policy_configs):
            print(f"Running config {config_idx + 1}/{len(policy_configs)}")
            
            episodes = []
            for episode in range(self.num_episodes):
                episode_data = self.run_episode(action_policy_fn, config)
                episodes.append(episode_data)
            
            results[f'config_{config_idx}'] = {
                'config': config,
                'episodes': episodes,
            }
        
        return results


class DataAggregator:
    
    def __init__(self, experiment_results: Dict[str, Dict]):
        self.results = experiment_results
    
    def aggregate_by_config(self) -> Dict[str, Dict[str, Any]]:
        aggregated = {}
        
        for config_name, result in self.results.items():
            episodes = result['episodes']
            
            all_rewards = {agent_type: [] for agent_type in episodes[0]['rewards'][0].keys()}
            all_market_prices = []
            
            for episode in episodes:
                rewards = episode['rewards']
                markets = episode['market_states']
                
                for step_rewards in rewards:
                    for agent_type, reward_values in step_rewards.items():
                        if isinstance(reward_values, np.ndarray):
                            all_rewards[agent_type].extend(reward_values)
                        else:
                            all_rewards[agent_type].append(reward_values)
                
                for market in markets:
                    if 'market_price' in market:
                        all_market_prices.append(market['market_price'])
                    elif 'price' in market:
                        all_market_prices.append(market['price'])
            
            convergence_rewards = {agent_type: [] for agent_type in all_rewards.keys()}
            for episode in episodes[-20:]:
                rewards = episode['rewards']
                for step_rewards in rewards[-25:]:
                    for agent_type, reward_values in step_rewards.items():
                        if isinstance(reward_values, np.ndarray):
                            convergence_rewards[agent_type].extend(reward_values)
                        else:
                            convergence_rewards[agent_type].append(reward_values)
            
            aggregated[config_name] = {
                'config': result['config'],
                'avg_rewards': {
                    agent_type: float(np.mean(rewards)) if rewards else 0.0
                    for agent_type, rewards in all_rewards.items()
                },
                'std_rewards': {
                    agent_type: float(np.std(rewards)) if rewards else 0.0
                    for agent_type, rewards in all_rewards.items()
                },
                'convergence_rewards': {
                    agent_type: float(np.mean(rewards)) if rewards else 0.0
                    for agent_type, rewards in convergence_rewards.items()
                },
                'avg_market_price': float(np.mean(all_market_prices)) if all_market_prices else 0.0,
            }
        
        return aggregated
    
    def get_agent_type_metrics(self, agent_type: str) -> Dict[str, List[float]]:
        metrics = {}
        
        for config_name, result in self.results.items():
            episodes = result['episodes']
            agent_rewards = []
            
            for episode in episodes:
                rewards = episode['rewards']
                for step_rewards in rewards:
                    if agent_type in step_rewards:
                        reward_values = step_rewards[agent_type]
                        if isinstance(reward_values, np.ndarray):
                            agent_rewards.extend(reward_values)
                        else:
                            agent_rewards.append(reward_values)
            
            metrics[config_name] = agent_rewards
        
        return metrics
