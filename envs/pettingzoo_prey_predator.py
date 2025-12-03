"""
Wrapper for PettingZoo's Simple Tag Environment.
This provides a standardized interface for the predator-prey game.
"""

from pettingzoo.mpe import simple_tag_v3
from typing import Optional, Dict, Any
import numpy as np


class SimpleTagEnv:
    """
    Wrapper for PettingZoo's Simple Tag environment.
    
    In Simple Tag:
    - Adversaries (predators) try to tag good agents (prey)
    - Good agents try to avoid being tagged
    - Environment includes landmarks and continuous actions
    """
    
    def __init__(
        self,
        n_adversaries: int = 3,
        n_good: int = 1,
        max_cycles: int = 200,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Simple Tag environment.
        
        Args:
            n_adversaries: Number of adversaries (predators)
            n_good: Number of good agents (prey)
            max_cycles: Maximum steps per episode
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        self.n_adversaries = n_adversaries
        self.n_good = n_good
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        
        # Create the PettingZoo environment
        self.env = simple_tag_v3.parallel_env(
            num_good=n_good,
            num_adversaries=n_adversaries,
            num_obstacles=2,
            max_cycles=max_cycles,
            continuous_actions=False,  # Use discrete actions
            render_mode=render_mode
        )
        
        # Get agent lists
        self.env.reset()
        self.possible_agents = self.env.possible_agents
        
        # Separate adversaries and good agents
        self.adversaries = [agent for agent in self.possible_agents if 'adversary' in agent]
        self.good_agents = [agent for agent in self.possible_agents if 'agent' in agent]
        
        # For compatibility with training scripts
        self.predators = self.adversaries  # Alias
        self.prey_agents = self.good_agents  # Alias
        
        self.agents = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents.copy()
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """Execute one step in the environment."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents.copy()
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment."""
        if self.render_mode is not None:
            return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self.env.observation_space(agent)
    
    def action_space(self, agent: str):
        """Get action space for an agent."""
        return self.env.action_space(agent)
    
    @property
    def num_adversaries(self):
        """Number of adversaries (predators)."""
        return self.n_adversaries
    
    @property
    def num_good(self):
        """Number of good agents (prey)."""
        return self.n_good


# Create an alias for backward compatibility
PreyPredatorEnv = SimpleTagEnv

