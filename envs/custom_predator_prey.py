"""
Custom Predator-Prey Gridworld Environment.
Clean implementation without PettingZoo dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces


class CustomPredatorPreyEnv:
    """
    Custom predator-prey environment with:
    - Discrete grid world
    - Fixed obstacles (non-moving)
    - Symmetric agent sizes
    - Simple reward structure
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        n_predators: int = 2,
        n_prey: int = 2,
        n_obstacles: int = 2,
        max_steps: int = 200,
        agent_radius: float = 0.5,      # Same size for both predator and prey
        obstacle_radius: float = 1.0,   # Double the agent radius
        capture_radius: float = 1.5,    # Distance to catch prey
        tag_reward: float = 10.0,       # Reward when caught
    ):
        """
        Initialize custom predator-prey environment.
        
        Args:
            grid_size: Size of square grid (grid_size × grid_size)
            n_predators: Number of predators
            n_prey: Number of prey
            n_obstacles: Number of fixed obstacles
            max_steps: Maximum steps per episode
            agent_radius: Radius for both predators and prey (SAME SIZE)
            obstacle_radius: Radius for obstacles (2× agent_radius)
            capture_radius: Distance for successful tag
            tag_reward: Reward magnitude when caught (same for both)
        """
        self.grid_size = grid_size
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.agent_radius = agent_radius
        self.obstacle_radius = obstacle_radius
        self.capture_radius = capture_radius
        self.tag_reward = tag_reward
        
        # Agent names
        self.adversaries = [f"predator_{i}" for i in range(n_predators)]
        self.good_agents = [f"prey_{i}" for i in range(n_prey)]
        self.agents = self.adversaries + self.good_agents
        
        # Positions (will be set in reset)
        self.predator_positions = None
        self.prey_positions = None
        self.obstacle_positions = None
        
        # State
        self.current_step = 0
        self.done = False
        
        # Observation and action spaces
        # Observation: [own_x, own_y, own_vx, own_vy, other_agents_x, other_agents_y, obstacles_x, obstacles_y]
        # Include velocity for compatibility with PettingZoo format
        obs_size = 2 + 2 + (n_predators + n_prey - 1) * 2 + n_obstacles * 2
        self._obs_space = spaces.Box(low=-10, high=10, shape=(obs_size,), dtype=np.float32)
        
        # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        self._action_space = spaces.Discrete(5)
        
        # For rendering
        self.fig = None
        self.ax = None
    
    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self._obs_space
    
    def action_space(self, agent: str):
        """Get action space for an agent."""
        return self._action_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Randomly place predators in top half
        self.predator_positions = np.random.uniform(
            low=[0, self.grid_size * 0.5],
            high=[self.grid_size, self.grid_size],
            size=(self.n_predators, 2)
        )
        
        # Randomly place prey in bottom half
        self.prey_positions = np.random.uniform(
            low=[0, 0],
            high=[self.grid_size, self.grid_size * 0.5],
            size=(self.n_prey, 2)
        )
        
        # Place FIXED obstacles in random positions
        self.obstacle_positions = np.random.uniform(
            low=self.grid_size * 0.2,
            high=self.grid_size * 0.8,
            size=(self.n_obstacles, 2)
        )
        
        self.current_step = 0
        self.done = False
        
        # Get observations for all agents
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        # Predator observations
        for i, agent in enumerate(self.adversaries):
            obs = self._get_agent_observation(
                self.predator_positions[i],
                self.predator_positions,
                self.prey_positions,
                is_predator=True
            )
            observations[agent] = obs
        
        # Prey observations
        for i, agent in enumerate(self.good_agents):
            obs = self._get_agent_observation(
                self.prey_positions[i],
                self.predator_positions,
                self.prey_positions,
                is_predator=False
            )
            observations[agent] = obs
        
        return observations
    
    def _get_agent_observation(
        self,
        own_pos: np.ndarray,
        predator_positions: np.ndarray,
        prey_positions: np.ndarray,
        is_predator: bool
    ) -> np.ndarray:
        """Get observation for a single agent."""
        obs = []
        
        # Own position (normalized to [0, 1])
        obs.extend(own_pos / self.grid_size)
        
        # Own velocity (for now, always 0 - could add later)
        obs.extend([0.0, 0.0])
        
        # All OTHER agents' positions (relative to self, normalized)
        all_positions = np.vstack([predator_positions, prey_positions])
        
        for pos in all_positions:
            # Skip self
            if np.allclose(pos, own_pos, atol=0.01):
                continue
            # Relative position (normalized)
            relative = (pos - own_pos) / self.grid_size
            obs.extend(relative)
        
        # Obstacle positions (relative to self, normalized)
        for obs_pos in self.obstacle_positions:
            relative = (obs_pos - own_pos) / self.grid_size
            obs.extend(relative)
        
        # Verify observation size is correct
        expected_size = 2 + 2 + (self.n_predators + self.n_prey - 1) * 2 + self.n_obstacles * 2
        obs_array = np.array(obs, dtype=np.float32)
        
        if len(obs_array) != expected_size:
            # Pad or trim to match expected size
            if len(obs_array) < expected_size:
                obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)))
            else:
                obs_array = obs_array[:expected_size]
        
        return obs_array
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary mapping agent names to action indices
        
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Move agents based on actions
        self._move_agents(actions)
        
        # Check collisions and compute rewards
        rewards = self._compute_rewards()
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        
        self.done = self.current_step >= self.max_steps
        
        # Get new observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _move_agents(self, actions: Dict[str, int]):
        """Move agents based on actions."""
        move_map = {
            0: np.array([0, 0]),      # Stay
            1: np.array([0, 1]),      # Up
            2: np.array([0, -1]),     # Down
            3: np.array([-1, 0]),     # Left
            4: np.array([1, 0])       # Right
        }
        
        # Move predators
        for i, agent in enumerate(self.adversaries):
            if agent in actions:
                action = actions[agent]
                movement = move_map[action] * 0.5  # Move 0.5 units per step
                new_pos = self.predator_positions[i] + movement
                
                # Check bounds
                new_pos = np.clip(new_pos, 0, self.grid_size)
                
                # Check obstacle collision
                if not self._collides_with_obstacle(new_pos):
                    self.predator_positions[i] = new_pos
        
        # Move prey
        for i, agent in enumerate(self.good_agents):
            if agent in actions:
                action = actions[agent]
                movement = move_map[action] * 0.5
                new_pos = self.prey_positions[i] + movement
                
                # Check bounds
                new_pos = np.clip(new_pos, 0, self.grid_size)
                
                # Check obstacle collision
                if not self._collides_with_obstacle(new_pos):
                    self.prey_positions[i] = new_pos
    
    def _collides_with_obstacle(self, position: np.ndarray) -> bool:
        """Check if position collides with any obstacle."""
        for obs_pos in self.obstacle_positions:
            distance = np.linalg.norm(position - obs_pos)
            if distance < (self.agent_radius + self.obstacle_radius):
                return True
        return False
    
    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute rewards for all agents.
        
        ENHANCED REWARD STRUCTURE (helps prey learn better):
        
        Predators:
        - Each tag: +10.0
        - Close to prey bonus: +0.2 per prey within 3 units (helps early learning)
        
        Prey:
        - Each tag: -10.0  
        - Survival bonus: +0.1 per step
        - Distance reward: +0.3 if all predators > 3 units away (encourages evasion)
        
        This gives prey more learning signal beyond just tags!
        
        Example (200 steps, good prey that escapes):
        - Prey staying far: (200 × 0.1) + (200 × 0.3) = 20 + 60 = +80 ✅
        - Prey tagged 2x but far otherwise: 80 - 20 = +60
        
        Win rule: Predator wins if predator_reward > 0 (caught prey)
        """
        rewards = {}
        
        # Initialize all to 0
        for agent in self.adversaries:
            rewards[agent] = 0.0
        for agent in self.good_agents:
            rewards[agent] = 0.0
        
        # Survival bonus for prey (base reward for being alive)
        for agent in self.good_agents:
            rewards[agent] += 0.1
        
        # Check distances and compute rewards
        for i, predator in enumerate(self.adversaries):
            for j, prey in enumerate(self.good_agents):
                distance = np.linalg.norm(
                    self.predator_positions[i] - self.prey_positions[j]
                )
                
                # Tag event
                if distance < self.capture_radius:
                    rewards[predator] += self.tag_reward   # +10
                    rewards[prey] -= self.tag_reward       # -10
                
                # Proximity bonus for predator (helps early learning)
                elif distance < 3.0:  # Within 3 units but not caught
                    rewards[predator] += 0.2
        
        # Distance reward for prey (encourages staying away from ALL predators)
        for j, prey_agent in enumerate(self.good_agents):
            min_distance_to_predator = min(
                np.linalg.norm(self.predator_positions[i] - self.prey_positions[j])
                for i in range(self.n_predators)
            )
            
            # Reward for maintaining distance from all predators
            if min_distance_to_predator > 3.0:
                rewards[prey_agent] += 0.3  # Good evasion!
        
        return rewards
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if self.fig is None:
            plt.ion()  # Interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'Predator-Prey Environment (Step {self.current_step}/{self.max_steps})')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Draw obstacles (FIXED, dark gray, double agent size)
        for obs_pos in self.obstacle_positions:
            circle = patches.Circle(
                obs_pos, radius=self.obstacle_radius, 
                facecolor='darkgray', 
                edgecolor='black', 
                linewidth=2,
                zorder=1,
                label='Obstacle' if obs_pos is self.obstacle_positions[0] else ""
            )
            self.ax.add_patch(circle)
        
        # Draw predators (red) - same size as prey now
        for i, pos in enumerate(self.predator_positions):
            circle = patches.Circle(
                pos, radius=self.agent_radius, 
                facecolor='red', 
                edgecolor='darkred', 
                linewidth=2,
                zorder=3,
                label='Predator' if i == 0 else ""
            )
            self.ax.add_patch(circle)
            # Add label
            self.ax.text(pos[0], pos[1], f'P{i}', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # Draw prey (green) - same size as predators now
        for i, pos in enumerate(self.prey_positions):
            circle = patches.Circle(
                pos, radius=self.agent_radius, 
                facecolor='limegreen', 
                edgecolor='darkgreen', 
                linewidth=2,
                zorder=3,
                label='Prey' if i == 0 else ""
            )
            self.ax.add_patch(circle)
            # Add label
            self.ax.text(pos[0], pos[1], f'p{i}', 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # Draw capture radius around predators (for visualization)
        for pos in self.predator_positions:
            circle = patches.Circle(
                pos, radius=self.capture_radius,
                facecolor='none',
                edgecolor='red',
                linestyle='--',
                linewidth=1,
                alpha=0.3,
                zorder=2
            )
            self.ax.add_patch(circle)
        
        # Legend
        self.ax.legend(loc='upper right', fontsize=10)
        
        plt.pause(0.001)
        plt.draw()
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    @property
    def num_adversaries(self):
        """Number of predators."""
        return self.n_predators
    
    @property
    def num_good(self):
        """Number of prey."""
        return self.n_prey


class SimpleTagEnv:
    """
    Wrapper to maintain compatibility with existing training scripts.
    This replaces the PettingZoo version.
    """
    
    def __init__(
        self,
        n_adversaries: int = 2,
        n_good: int = 2,
        max_cycles: int = 200,
        grid_size: int = 20,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment (compatible with existing code).
        
        Args:
            n_adversaries: Number of predators
            n_good: Number of prey
            max_cycles: Maximum steps per episode
            grid_size: Size of grid world
            render_mode: "human" for visual rendering, None for no rendering
        """
        self.n_adversaries = n_adversaries
        self.n_good = n_good
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        
        # Create internal environment
        self.env = CustomPredatorPreyEnv(
            grid_size=grid_size,
            n_predators=n_adversaries,
            n_prey=n_good,
            n_obstacles=3,
            max_steps=max_cycles,
            capture_radius=1.5,
            tag_reward=10.0,
            step_penalty=-0.1,
            survival_bonus=0.1
        )
        
        # Agent lists (for compatibility)
        self.adversaries = self.env.adversaries
        self.good_agents = self.env.good_agents
        self.agents = self.env.agents
        
        # Aliases
        self.predators = self.adversaries
        self.prey_agents = self.good_agents
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        obs, infos = self.env.reset(seed=seed), {}
        self.agents = self.env.agents.copy()
        return obs, infos
    
    def step(self, actions: Dict[str, int]):
        """Execute one step."""
        return self.env.step(actions)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.env.render()
    
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
        """Number of predators."""
        return self.n_adversaries
    
    @property
    def num_good(self):
        """Number of prey."""
        return self.n_good


# Backward compatibility alias
PreyPredatorEnv = SimpleTagEnv

