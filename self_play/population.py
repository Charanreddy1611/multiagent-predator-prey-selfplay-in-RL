"""
Population-Based Self-Play Strategy.
Maintain a population of diverse agents and sample opponents from it.
"""

import copy
import random
from typing import List, Dict, Any


class PopulationSelfPlay:
    """
    Population-based self-play that maintains diverse agent populations
    for both predators and prey.
    """
    
    def __init__(
        self,
        predator_agent,
        prey_agent,
        population_size: int = 5,
        update_interval: int = 500,
        sampling_strategy: str = "uniform"  # "uniform", "recent", "fitness"
    ):
        """
        Initialize population-based self-play.
        
        Args:
            predator_agent: Initial predator agent
            prey_agent: Initial prey agent
            population_size: Maximum size of each population
            update_interval: Steps between adding agents to population
            sampling_strategy: How to sample opponents from population
        """
        self.predator_agent = predator_agent
        self.prey_agent = prey_agent
        self.population_size = population_size
        self.update_interval = update_interval
        self.sampling_strategy = sampling_strategy
        
        # Initialize populations with current agents
        self.predator_population = [copy.deepcopy(predator_agent)]
        self.prey_population = [copy.deepcopy(prey_agent)]
        
        # Track fitness/performance of population members
        self.predator_fitness = [0.0]
        self.prey_fitness = [0.0]
        
        self.steps = 0
        self.current_predator_opponent_idx = 0
        self.current_prey_opponent_idx = 0
    
    def should_update_population(self) -> bool:
        """Check if it's time to add current agents to population."""
        return self.steps % self.update_interval == 0 and self.steps > 0
    
    def add_to_population(self):
        """Add current agents to their respective populations."""
        # Add predator
        self.predator_population.append(copy.deepcopy(self.predator_agent))
        self.predator_fitness.append(0.0)
        
        # Add prey
        self.prey_population.append(copy.deepcopy(self.prey_agent))
        self.prey_fitness.append(0.0)
        
        # Maintain population size limit
        if len(self.predator_population) > self.population_size:
            # Remove worst performer
            worst_idx = self.predator_fitness.index(min(self.predator_fitness))
            self.predator_population.pop(worst_idx)
            self.predator_fitness.pop(worst_idx)
        
        if len(self.prey_population) > self.population_size:
            worst_idx = self.prey_fitness.index(min(self.prey_fitness))
            self.prey_population.pop(worst_idx)
            self.prey_fitness.pop(worst_idx)
        
        print(f"Step {self.steps}: Population updated. "
              f"Predators: {len(self.predator_population)}, Prey: {len(self.prey_population)}")
    
    def sample_opponent(self, population: List, fitness: List[float]) -> int:
        """Sample an opponent from population based on strategy."""
        if self.sampling_strategy == "uniform":
            return random.randint(0, len(population) - 1)
        
        elif self.sampling_strategy == "recent":
            # Sample more recent agents with higher probability
            weights = [i + 1 for i in range(len(population))]
            return random.choices(range(len(population)), weights=weights)[0]
        
        elif self.sampling_strategy == "fitness":
            # Sample based on fitness (higher fitness = higher prob)
            # Shift fitness values to be positive (add minimum + 1)
            min_fitness = min(fitness)
            if min_fitness < 0:
                # Shift all values to be positive
                shifted_fitness = [f - min_fitness + 1.0 for f in fitness]
            else:
                shifted_fitness = [f + 1.0 for f in fitness]  # Add 1 to avoid zero weights
            
            # Check if sum is still valid
            total = sum(shifted_fitness)
            if total <= 0:
                return random.randint(0, len(population) - 1)
            
            return random.choices(range(len(population)), weights=shifted_fitness)[0]
        
        else:
            return random.randint(0, len(population) - 1)
    
    def step(self, observations, deterministic=False):
        """Select actions using current learner vs population opponents."""
        self.steps += 1
        
        if self.should_update_population():
            self.add_to_population()
            # Sample new opponents
            self.current_predator_opponent_idx = self.sample_opponent(
                self.predator_population, self.predator_fitness
            )
            self.current_prey_opponent_idx = self.sample_opponent(
                self.prey_population, self.prey_fitness
            )
        
        # Current learner actions
        predator_actions = self.predator_agent.select_actions(
            observations['predators'],
            deterministic=deterministic
        )
        prey_actions = self.prey_agent.select_actions(
            observations['prey'],
            deterministic=deterministic
        )
        
        return {
            'predators': predator_actions,
            'prey': prey_actions
        }
    
    def update(self):
        """Update both predator and prey agents."""
        predator_metrics = self.predator_agent.update()
        prey_metrics = self.prey_agent.update()
        
        return {
            'predator': predator_metrics,
            'prey': prey_metrics
        }
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transitions for both agents."""
        self.predator_agent.store_transition(
            obs['predators'],
            actions['predators'],
            rewards['predators'],
            next_obs['predators'],
            dones['predators']
        )
        
        self.prey_agent.store_transition(
            obs['prey'],
            actions['prey'],
            rewards['prey'],
            next_obs['prey'],
            dones['prey']
        )
        
        # Update fitness based on rewards
        avg_predator_reward = sum(rewards['predators']) / len(rewards['predators'])
        avg_prey_reward = sum(rewards['prey']) / len(rewards['prey'])
        
        self.predator_fitness[-1] += avg_predator_reward
        self.prey_fitness[-1] += avg_prey_reward
    
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        return {
            'total_steps': self.steps,
            'predator_population_size': len(self.predator_population),
            'prey_population_size': len(self.prey_population),
            'avg_predator_fitness': sum(self.predator_fitness) / len(self.predator_fitness),
            'avg_prey_fitness': sum(self.prey_fitness) / len(self.prey_fitness)
        }

