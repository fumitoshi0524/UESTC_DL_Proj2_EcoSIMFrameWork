import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json


class ExperimentVisualizer:
    
    def __init__(self, results: Dict[str, Dict[str, Any]], output_dir: str = None):
        self.results = results
        self.output_dir = output_dir or r'f:\projects\production_game\examples\government_intervention\results'
    
    def plot_reward_comparison(self, agent_type: str, filename: str = None):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        intervention_levels = []
        avg_rewards = []
        std_rewards = []
        
        for config_name in sorted(self.results.keys()):
            config_data = self.results[config_name]
            intervention = config_data['config'].get('intervention_strength', 0.0)
            
            if agent_type in config_data['avg_rewards']:
                intervention_levels.append(intervention)
                avg_rewards.append(config_data['avg_rewards'][agent_type])
                std_rewards.append(config_data['std_rewards'][agent_type])
        
        if not intervention_levels:
            print(f"No data for agent type: {agent_type}")
            return
        
        ax.errorbar(
            intervention_levels,
            avg_rewards,
            yerr=std_rewards,
            marker='o',
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            label=agent_type,
        )
        
        ax.set_xlabel('Intervention Strength', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(f'{agent_type.capitalize()} Reward vs Intervention Strength', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if filename:
            plt.savefig(f'{self.output_dir}/{filename}', dpi=100, bbox_inches='tight')
            print(f"Saved: {self.output_dir}/{filename}")
        
        plt.close()
    
    def plot_all_agents_comparison(self, filename: str = 'all_agents_comparison.png'):
        
        agent_types = set()
        for config_data in self.results.values():
            agent_types.update(config_data['avg_rewards'].keys())
        
        fig, axes = plt.subplots(1, len(agent_types), figsize=(6 * len(agent_types), 5))
        if len(agent_types) == 1:
            axes = [axes]
        
        for idx, agent_type in enumerate(sorted(agent_types)):
            ax = axes[idx]
            
            intervention_levels = []
            avg_rewards = []
            std_rewards = []
            
            for config_name in sorted(self.results.keys()):
                config_data = self.results[config_name]
                intervention = config_data['config'].get('intervention_strength', 0.0)
                
                if agent_type in config_data['avg_rewards']:
                    intervention_levels.append(intervention)
                    avg_rewards.append(config_data['avg_rewards'][agent_type])
                    std_rewards.append(config_data['std_rewards'][agent_type])
            
            ax.errorbar(
                intervention_levels,
                avg_rewards,
                yerr=std_rewards,
                marker='o',
                markersize=8,
                capsize=5,
                capthick=2,
                linewidth=2,
            )
            
            ax.set_xlabel('Intervention Strength', fontsize=11)
            ax.set_ylabel('Average Reward', fontsize=11)
            ax.set_title(f'{agent_type.capitalize()}', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=100, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/{filename}")
        plt.close()
    
    def plot_market_price_evolution(self, filename: str = 'market_price_evolution.png'):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for config_name in sorted(self.results.keys()):
            config_data = self.results[config_name]
            intervention = config_data['config'].get('intervention_strength', 0.0)
            price = config_data.get('avg_market_price', 0.0)
            
            ax.scatter(intervention, price, s=100, label=f'Level {intervention:.2f}')
        
        ax.set_xlabel('Intervention Strength', fontsize=12)
        ax.set_ylabel('Average Market Price', fontsize=12)
        ax.set_title('Market Price vs Intervention Strength', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.savefig(f'{self.output_dir}/{filename}', dpi=100, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/{filename}")
        plt.close()
    
    def plot_convergence_comparison(self, filename: str = 'convergence_comparison.png'):
        
        agent_types = set()
        for config_data in self.results.values():
            agent_types.update(config_data['convergence_rewards'].keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(self.results))
        width = 0.2
        
        for idx, agent_type in enumerate(sorted(agent_types)):
            convergence_rewards = []
            
            for config_name in sorted(self.results.keys()):
                config_data = self.results[config_name]
                if agent_type in config_data['convergence_rewards']:
                    convergence_rewards.append(config_data['convergence_rewards'][agent_type])
            
            ax.bar(x_pos + idx * width, convergence_rewards, width, label=agent_type)
        
        ax.set_xlabel('Intervention Level', fontsize=12)
        ax.set_ylabel('Convergence Reward', fontsize=12)
        ax.set_title('Convergence Rewards by Intervention Level', fontsize=14)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(sorted(self.results.keys()), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=100, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/{filename}")
        plt.close()
    
    def generate_all_plots(self):
        
        agent_types = set()
        for config_data in self.results.values():
            agent_types.update(config_data['avg_rewards'].keys())
        
        print("Generating plots...")
        
        for agent_type in agent_types:
            filename = f'{agent_type}_reward_comparison.png'
            self.plot_reward_comparison(agent_type, filename)
        
        self.plot_all_agents_comparison()
        self.plot_market_price_evolution()
        self.plot_convergence_comparison()
        
        print("All plots generated successfully")


def load_results_json(filepath: str) -> Dict[str, Dict[str, Any]]:
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = {}
    for key, value in data.items():
        results[key] = value
    
    return results


def main():
    
    results_file = r'f:\projects\production_game\examples\government_intervention\results\experiment_results.json'
    
    try:
        results = load_results_json(results_file)
        visualizer = ExperimentVisualizer(results)
        visualizer.generate_all_plots()
        print("Visualization complete")
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        print("Please run the experiment first with run_experiment.py")


if __name__ == '__main__':
    main()
