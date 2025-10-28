import numpy as np
from typing import Dict, Any
import sys
sys.path.append(r'f:\projects\production_game')

from framework.core import (
    AgentSpec,
    MarketSpec,
    GameEnvironment,
)


def farmer_reward_fn(
    actions: np.ndarray,
    market: Dict[str, Any],
    policy: Dict[str, Any],
) -> np.ndarray:
    c0 = 0.2
    c1 = 0.3
    
    num_farmers = actions.shape[0] if len(actions.shape) > 1 else 1
    
    if 'market_price' not in market:
        return np.zeros(num_farmers)
    
    price = market['market_price']
    intervention_strength = policy.get('intervention_strength', 0.0)
    price_floor = market.get('price_floor', 0.0)
    
    supply = actions.flatten() if len(actions.shape) > 1 else actions
    if len(supply) < num_farmers:
        supply = np.tile(supply, (num_farmers // len(supply) + 1))[:num_farmers]
    
    revenue = price * supply
    
    if intervention_strength > 0:
        support = (price_floor - price) * supply * min(1.0, intervention_strength * 2)
        revenue += support
    
    cost = c0 * supply + c1 * supply * supply
    
    profit = np.maximum(-0.1, revenue - cost)
    
    return np.clip(profit, -1.0, 1.0)


def consumer_reward_fn(
    actions: np.ndarray,
    market: Dict[str, Any],
    policy: Dict[str, Any],
) -> np.ndarray:
    
    num_consumers = actions.shape[0] if len(actions.shape) > 1 else 1
    
    if 'market_price' not in market:
        return np.zeros(num_consumers)
    
    price = market['market_price']
    supply = market.get('total_supply', 0.5)
    intervention_strength = policy.get('intervention_strength', 0.0)
    
    demand = actions.flatten() if len(actions.shape) > 1 else actions
    if len(demand) < num_consumers:
        demand = np.tile(demand, (num_consumers // len(demand) + 1))[:num_consumers]
    
    quantity = demand * 0.5 + 0.5
    satisfaction = np.log(quantity + 1e-6) * 0.5 + 0.5
    
    cost = price * quantity
    utility = satisfaction - cost * 0.5
    
    if intervention_strength > 0:
        subsidy = supply * 0.01 * intervention_strength
        utility += subsidy * 0.3
    
    return np.clip(utility, -1.0, 1.0)


def government_reward_fn(
    actions: np.ndarray,
    market: Dict[str, Any],
    policy: Dict[str, Any],
) -> np.ndarray:
    
    num_gov_agents = actions.shape[0] if len(actions.shape) > 1 else 1
    
    intervention_strength = policy.get('intervention_strength', 0.0)
    
    total_supply = market.get('total_supply', 0.5)
    total_demand = market.get('total_demand', 0.5)
    price = market.get('market_price', 0.5)
    
    price_stability = 1.0 - np.abs(price - 0.5) / 0.5
    
    supply_demand_balance = 1.0 - np.abs(total_supply - total_demand) / max(total_supply, total_demand, 0.1)
    
    cost_of_intervention = intervention_strength * intervention_strength * 0.5
    
    welfare = price_stability * 0.4 + supply_demand_balance * 0.4 - cost_of_intervention * 0.2
    
    return np.ones(num_gov_agents) * np.clip(welfare, -1.0, 1.0)


def market_clearing_fn(
    actions: Dict[str, np.ndarray],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    p0 = 0.5
    alpha = 0.5
    
    farmer_actions = actions.get('farmer', np.array([[0.5]]))
    consumer_actions = actions.get('consumer', np.array([[0.5]]))
    gov_actions = actions.get('government', np.array([[0.5]]))
    
    if isinstance(farmer_actions, np.ndarray) and farmer_actions.size > 0:
        if len(farmer_actions.shape) > 1:
            total_supply = float(np.mean(farmer_actions))
        else:
            total_supply = float(np.mean(farmer_actions))
    else:
        total_supply = 0.5
    
    if isinstance(consumer_actions, np.ndarray) and consumer_actions.size > 0:
        if len(consumer_actions.shape) > 1:
            total_demand = float(np.mean(consumer_actions))
        else:
            total_demand = float(np.mean(consumer_actions))
    else:
        total_demand = 0.5
    
    total_supply = np.clip(total_supply, 0.1, 1.0)
    total_demand = np.clip(total_demand, 0.1, 1.0)
    
    intervention_strength = policy.get('intervention_strength', 0.0)
    price_floor = policy.get('price_floor', 0.4)
    
    eq_price = p0 + alpha * (total_demand - total_supply)
    
    if intervention_strength > 0 and total_supply < total_demand:
        price = max(eq_price, price_floor * (1 + intervention_strength * 0.2))
    else:
        price = eq_price
    
    price = float(np.clip(price, 0.2, 0.8))
    
    return {
        'total_supply': total_supply,
        'total_demand': total_demand,
        'market_price': price,
        'price_floor': price_floor,
        'intervention_strength': intervention_strength,
    }


def create_government_intervention_env(
    intervention_strength: float,
    num_episodes: int = 1,
    episode_length: int = 100,
) -> GameEnvironment:
    
    farmer_spec = AgentSpec(
        name='farmer',
        num_agents=3,
        obs_space=(4,),
        action_space=(3, 1),
        reward_fn=farmer_reward_fn,
        action_processor=lambda a: np.clip(a, 0, 1),
    )
    
    consumer_spec = AgentSpec(
        name='consumer',
        num_agents=2,
        obs_space=(4,),
        action_space=(2, 1),
        reward_fn=consumer_reward_fn,
        action_processor=lambda a: np.clip(a, 0, 1),
    )
    
    government_spec = AgentSpec(
        name='government',
        num_agents=1,
        obs_space=(5,),
        action_space=(1, 1),
        reward_fn=government_reward_fn,
        action_processor=lambda a: a,
    )
    
    market_spec = MarketSpec(
        clearing_fn=market_clearing_fn,
    )
    
    policy_state = {
        'intervention_strength': float(intervention_strength),
        'price_floor': 0.4 + intervention_strength * 0.1,
    }
    
    agents_specs = {
        'farmer': farmer_spec,
        'consumer': consumer_spec,
        'government': government_spec,
    }
    
    env = GameEnvironment(
        agents_specs=agents_specs,
        market_spec=market_spec,
        time_steps=episode_length,
    )
    
    env.policy_state = policy_state
    
    return env
