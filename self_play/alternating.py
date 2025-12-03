"""
Alternating Self-Play Strategy.
Train predators and prey in alternating phases.
"""

import copy
from typing import Dict, Any


class AlternatingSelfPlay:
    """
    Alternating self-play where predators and prey take turns being frozen
    while the other group trains.
    """
    
    def __init__(
        self,
        predator_agent,
        prey_agent,
        switch_interval: int = 1000,  # Steps before switching
    ):
        """
        Initialize alternating self-play.
        
        Args:
            predator_agent: Predator RL agent
            prey_agent: Prey RL agent
            switch_interval: Number of steps before switching active learner
        """
        self.predator_agent = predator_agent
        self.prey_agent = prey_agent
        self.switch_interval = switch_interval
        
        self.steps = 0
        self.training_predators = True  # Start by training predators
        self.history = []
    
    def should_switch(self) -> bool:
        """Check if it's time to switch training focus."""
        return self.steps % self.switch_interval == 0 and self.steps > 0
    
    def switch_training_focus(self):
        """Switch between training predators and prey."""
        self.training_predators = not self.training_predators
        
        # Save snapshot of opponent
        if self.training_predators:
            opponent_snapshot = copy.deepcopy(self.prey_agent)
            print(f"Step {self.steps}: Switching to train PREDATORS")
        else:
            opponent_snapshot = copy.deepcopy(self.predator_agent)
            print(f"Step {self.steps}: Switching to train PREY")
        
        self.history.append({
            'step': self.steps,
            'training_predators': self.training_predators,
            'opponent_snapshot': opponent_snapshot
        })
    
    def get_active_agents(self):
        """Get which agents are actively training."""
        return {
            'predators_active': self.training_predators,
            'prey_active': not self.training_predators
        }
    
    def step(self, observations, deterministic=False):
        """
        Select actions for both predators and prey.
        
        Args:
            observations: Dict with 'predators' and 'prey' observations
            deterministic: Whether to use deterministic actions
        
        Returns:
            Dict with 'predators' and 'prey' actions
        """
        self.steps += 1
        
        if self.should_switch():
            self.switch_training_focus()
        
        # Get actions from both agents
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
        """Update the active learner."""
        if self.training_predators:
            return self.predator_agent.update()
        else:
            return self.prey_agent.update()
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transitions for the active learner."""
        if self.training_predators:
            self.predator_agent.store_transition(
                obs['predators'], 
                actions['predators'],
                rewards['predators'],
                next_obs['predators'],
                dones['predators']
            )
        else:
            self.prey_agent.store_transition(
                obs['prey'],
                actions['prey'],
                rewards['prey'],
                next_obs['prey'],
                dones['prey']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.steps,
            'training_predators': self.training_predators,
            'num_switches': len(self.history)
        }

