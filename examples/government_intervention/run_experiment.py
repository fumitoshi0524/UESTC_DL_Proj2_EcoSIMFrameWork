import numpy as np
from typing import Dict, List, Any
import json
import sys
sys.path.append(r'f:\projects\production_game')

from framework.core import ExperimentRunner, DataAggregator
from examples.government_intervention.scenario import create_government_intervention_env


class GovernmentInterventionExperiment:
    
    def __init__(
        self,
        intervention_levels: List[float] = None,
        num_episodes: int = 100,
        episode_length: int = 100,
    ):
        self.intervention_levels = intervention_levels or [0.0, 0.05, 0.1, 0.15, 0.2]
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.results = {}
    
    def _env_factory(self, intervention_strength: float):
        def factory():
            return create_government_intervention_env(
                intervention_strength=intervention_strength,
                episode_length=self.episode_length,
            )
        return factory
    
    def _policy_fn(self, observations: Dict[str, np.ndarray], step: int, explore_rate: float) -> Dict[str, np.ndarray]:
        
        actions = {}
        
        if 'farmer' in observations:
            obs = observations['farmer']
            if len(obs.shape) > 1:
                num_farmers = obs.shape[0]
                actions['farmer'] = np.random.uniform(0.3, 0.8, (num_farmers, 1)).astype(np.float32)
            else:
                actions['farmer'] = np.array([[np.random.uniform(0.3, 0.8)]], dtype=np.float32)
        
        if 'consumer' in observations:
            obs = observations['consumer']
            if len(obs.shape) > 1:
                num_consumers = obs.shape[0]
                actions['consumer'] = np.random.uniform(0.2, 0.9, (num_consumers, 1)).astype(np.float32)
            else:
                actions['consumer'] = np.array([[np.random.uniform(0.2, 0.9)]], dtype=np.float32)
        
        if 'government' in observations:
            obs = observations['government']
            if len(obs.shape) > 1:
                num_gov = obs.shape[0]
                actions['government'] = np.random.uniform(0.4, 0.7, (num_gov, 1)).astype(np.float32)
            else:
                actions['government'] = np.array([[np.random.uniform(0.4, 0.7)]], dtype=np.float32)
        
        return actions
    
    def run(self) -> Dict[str, Any]:
        
        print(f"Running Government Intervention Experiment")
        print(f"Intervention levels: {self.intervention_levels}")
        print(f"Episodes per level: {self.num_episodes}")
        print(f"Episode length: {self.episode_length}")
        print()
        
        all_results = {}
        
        for intervention_level in self.intervention_levels:
            print(f"Running intervention level: {intervention_level:.2f}")
            
            runner = ExperimentRunner(
                env_factory=self._env_factory(intervention_level),
                num_episodes=self.num_episodes,
                max_steps=self.episode_length,
            )
            
            config = {'intervention_strength': intervention_level}
            episodes = []
            
            for episode_idx in range(self.num_episodes):
                episode_data = runner.run_episode(self._policy_fn, config)
                episodes.append(episode_data)
                
                if (episode_idx + 1) % 20 == 0:
                    print(f"  Completed {episode_idx + 1}/{self.num_episodes} episodes")
            
            all_results[f'intervention_{intervention_level:.2f}'] = {
                'config': {'intervention_strength': intervention_level},
                'episodes': episodes,
            }
        
        aggregator = DataAggregator(all_results)
        self.results = aggregator.aggregate_by_config()
        
        return self.results
    
    def save_results(self, output_path: str):
        
        summary = {}
        for config_name, config_data in self.results.items():
            summary[config_name] = {
                'config': config_data['config'],
                'avg_rewards': config_data['avg_rewards'],
                'std_rewards': config_data['std_rewards'],
                'convergence_rewards': config_data['convergence_rewards'],
                'avg_market_price': config_data['avg_market_price'],
            }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    experiment = GovernmentInterventionExperiment(
        intervention_levels=[0.0, 0.05, 0.1, 0.15, 0.2],
        num_episodes=50,
        episode_length=100,
    )
    
    results = experiment.run()
    
    output_file = r'f:\projects\production_game\examples\government_intervention\results\experiment_results.json'
    experiment.save_results(output_file)
    
    print("\nExperiment Summary:")
    print("=" * 80)
    for config_name, config_data in results.items():
        print(f"\n{config_name}:")
        print(f"  Intervention: {config_data['config']['intervention_strength']:.2f}")
        print(f"  Average Rewards:")
        for agent_type, reward in config_data['avg_rewards'].items():
            print(f"    {agent_type}: {reward:.4f} ± {config_data['std_rewards'][agent_type]:.4f}")
        print(f"  Market Price: {config_data['avg_market_price']:.4f}")
