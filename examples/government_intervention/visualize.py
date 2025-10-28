import json
import sys
sys.path.append(r'f:\projects\production_game')

from framework.utils import ExperimentVisualizer

results = json.load(open(r'f:\projects\production_game\examples\government_intervention\results\experiment_results.json'))
output_dir = r'f:\projects\production_game\examples\government_intervention\results'

v = ExperimentVisualizer(results, output_dir)

v.plot_reward_comparison('farmer', 'farmer_rewards.png')
v.plot_reward_comparison('consumer', 'consumer_rewards.png')
v.plot_reward_comparison('government', 'government_rewards.png')
v.plot_all_agents_comparison('all_agents.png')
v.plot_market_price_evolution('market_price.png')
v.plot_convergence_comparison('convergence.png')

print("All plots generated successfully!")
