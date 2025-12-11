"""
Custom Predator-Prey Environment.
Replaced PettingZoo with custom implementation due to:
- Moving obstacles issue
- Collision detection problems
- Better control over game mechanics
"""

from envs.custom_predator_prey import CustomPredatorPreyEnv
from typing import Optional, Dict, Any
import numpy as np


class SimpleTagEnv:
    """
    Custom predator-prey environment.
    
    Features:
    - Fixed obstacles (non-moving)
    - Clear collision detection
    - Discrete grid world
    - Configurable capture radius
    - Better control over game mechanics
    """
    
    def __init__(
        self,
        n_adversaries: int = 2,
        n_good: int = 2,
        max_cycles: int = 200,
        grid_size: int = 20,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize custom predator-prey environment.
        
        Args:
            n_adversaries: Number of adversaries (predators)
            n_good: Number of good agents (prey)
            max_cycles: Maximum steps per episode
            grid_size: Size of grid (default 20×20)
            render_mode: Rendering mode ("human" or None)
        """
        self.n_adversaries = n_adversaries
        self.n_good = n_good
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.grid_size = grid_size
        
        # Create the custom environment
        self.env = CustomPredatorPreyEnv(
            grid_size=grid_size,
            n_predators=n_adversaries,
            n_prey=n_good,
            n_obstacles=2,
            max_steps=max_cycles,
            agent_radius=0.5,      # Same size for predators and prey
            obstacle_radius=1.0,   # Double the agent size
            capture_radius=1.5,    # Distance to catch
            tag_reward=10.0        # Symmetric reward magnitude
        )
        
        # Get agent lists
        self.adversaries = self.env.adversaries
        self.good_agents = self.env.good_agents
        
        # For compatibility with training scripts
        self.predators = self.adversaries  # Alias
        self.prey_agents = self.good_agents  # Alias
        
        self.agents = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        observations, infos = self.env.reset(seed=seed)
        self.agents = self.env.agents.copy()
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """Execute one step in the environment."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents.copy()
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
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

