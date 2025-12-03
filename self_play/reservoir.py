"""
Reservoir Self-Play Strategy.
Maintains a reservoir of historical opponents sampled throughout training.
"""

import copy
import random
from typing import List, Dict, Any
from collections import deque


class ReservoirSelfPlay:
    """
    Reservoir-based self-play that maintains a fixed-size reservoir
    of historical opponent snapshots using reservoir sampling.
    """
    
    def __init__(
        self,
        predator_agent,
        prey_agent,
        reservoir_size: int = 10,
        snapshot_interval: int = 500,
        sampling_temperature: float = 1.0,
        use_prioritized: bool = False
    ):
        """
        Initialize reservoir self-play.
        
        Args:
            predator_agent: Initial predator agent
            prey_agent: Initial prey agent
            reservoir_size: Maximum size of reservoir
            snapshot_interval: Steps between taking snapshots
            sampling_temperature: Temperature for sampling (higher = more uniform)
            use_prioritized: Use prioritized sampling based on recency
        """
        self.predator_agent = predator_agent
        self.prey_agent = prey_agent
        self.reservoir_size = reservoir_size
        self.snapshot_interval = snapshot_interval
        self.sampling_temperature = sampling_temperature
        self.use_prioritized = use_prioritized
        
        # Reservoirs for opponents
        self.predator_reservoir = []
        self.prey_reservoir = []
        
        # Track insertion order for prioritized sampling
        self.predator_timestamps = []
        self.prey_timestamps = []
        
        self.steps = 0
        self.snapshots_taken = 0
        
        # Current opponent indices
        self.current_predator_opponent_idx = None
        self.current_prey_opponent_idx = None
    
    def should_snapshot(self) -> bool:
        """Check if it's time to take a snapshot."""
        return self.steps % self.snapshot_interval == 0 and self.steps > 0
    
    def add_to_reservoir(self, agent, reservoir: List, timestamps: List):
        """
        Add agent to reservoir using reservoir sampling algorithm.
        
        Args:
            agent: Agent to add
            reservoir: Target reservoir
            timestamps: Timestamp list for reservoir
        """
        snapshot = copy.deepcopy(agent)
        
        if len(reservoir) < self.reservoir_size:
            # Reservoir not full, just add
            reservoir.append(snapshot)
            timestamps.append(self.snapshots_taken)
        else:
            # Reservoir full, use reservoir sampling
            # Random index to potentially replace
            idx = random.randint(0, self.snapshots_taken)
            if idx < self.reservoir_size:
                reservoir[idx] = snapshot
                timestamps[idx] = self.snapshots_taken
    
    def take_snapshots(self):
        """Take snapshots of current agents and add to reservoirs."""
        self.snapshots_taken += 1
        
        self.add_to_reservoir(
            self.predator_agent, 
            self.predator_reservoir,
            self.predator_timestamps
        )
        self.add_to_reservoir(
            self.prey_agent,
            self.prey_reservoir,
            self.prey_timestamps
        )
        
        print(f"Step {self.steps}: Snapshot #{self.snapshots_taken} taken. "
              f"Reservoir sizes - Predators: {len(self.predator_reservoir)}, "
              f"Prey: {len(self.prey_reservoir)}")
    
    def sample_from_reservoir(self, reservoir: List, timestamps: List) -> int:
        """
        Sample an opponent index from reservoir.
        
        Args:
            reservoir: Reservoir to sample from
            timestamps: Timestamps for each reservoir entry
        
        Returns:
            Index of sampled opponent
        """
        if len(reservoir) == 0:
            return None
        
        if not self.use_prioritized:
            # Uniform sampling
            return random.randint(0, len(reservoir) - 1)
        else:
            # Prioritized sampling (more recent = higher probability)
            # Use softmax with temperature
            max_time = max(timestamps) if timestamps else 1
            scores = [(t / max_time) ** (1.0 / self.sampling_temperature) 
                     for t in timestamps]
            total = sum(scores)
            weights = [s / total for s in scores]
            
            return random.choices(range(len(reservoir)), weights=weights)[0]
    
    def get_opponent_agent(self, reservoir: List, idx: int, current_agent):
        """Get opponent agent from reservoir or use current if reservoir empty."""
        if idx is None or len(reservoir) == 0:
            return current_agent
        return reservoir[idx]
    
    def step(self, observations, deterministic=False):
        """Select actions using current agents and reservoir opponents."""
        self.steps += 1
        
        if self.should_snapshot():
            self.take_snapshots()
            # Resample opponents after snapshot
            self.current_predator_opponent_idx = self.sample_from_reservoir(
                self.predator_reservoir, self.predator_timestamps
            )
            self.current_prey_opponent_idx = self.sample_from_reservoir(
                self.prey_reservoir, self.prey_timestamps
            )
        
        # Current learners play
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
        """Update both agents."""
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reservoir statistics."""
        return {
            'total_steps': self.steps,
            'snapshots_taken': self.snapshots_taken,
            'predator_reservoir_size': len(self.predator_reservoir),
            'prey_reservoir_size': len(self.prey_reservoir),
            'current_predator_opponent': self.current_predator_opponent_idx,
            'current_prey_opponent': self.current_prey_opponent_idx
        }

