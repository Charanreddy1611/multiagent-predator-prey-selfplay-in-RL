"""
Metrics tracking for multi-agent reinforcement learning.
"""

import json
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class MetricsTracker:
    """Track and analyze training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
        self.episode_data = []
    
    def log_episode(
        self,
        episode: int,
        predator_reward: float,
        prey_reward: float,
        episode_length: int,
        **kwargs
    ):
        """
        Log metrics for an episode.
        
        Args:
            episode: Episode number
            predator_reward: Average predator reward
            prey_reward: Average prey reward
            episode_length: Episode length
            **kwargs: Additional metrics
        """
        episode_info = {
            'episode': episode,
            'predator_reward': predator_reward,
            'prey_reward': prey_reward,
            'episode_length': episode_length,
            **kwargs
        }
        
        self.episode_data.append(episode_info)
        
        # Update running metrics
        self.metrics['predator_rewards'].append(predator_reward)
        self.metrics['prey_rewards'].append(prey_reward)
        self.metrics['episode_lengths'].append(episode_length)
        
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get statistics over recent episodes.
        
        Args:
            window: Number of recent episodes to consider
        
        Returns:
            Dictionary of statistics
        """
        if len(self.episode_data) == 0:
            return {}
        
        recent_data = self.episode_data[-window:]
        
        stats = {
            'avg_predator_reward': np.mean([d['predator_reward'] for d in recent_data]),
            'avg_prey_reward': np.mean([d['prey_reward'] for d in recent_data]),
            'avg_episode_length': np.mean([d['episode_length'] for d in recent_data]),
            'std_predator_reward': np.std([d['predator_reward'] for d in recent_data]),
            'std_prey_reward': np.std([d['prey_reward'] for d in recent_data]),
        }
        
        return stats
    
    def get_all_metrics(self) -> Dict[str, List]:
        """Get all tracked metrics."""
        return dict(self.metrics)
    
    def get_episode_data(self) -> List[Dict]:
        """Get all episode data."""
        return self.episode_data
    
    def compute_moving_average(self, metric_name: str, window: int = 100) -> List[float]:
        """
        Compute moving average for a metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size for moving average
        
        Returns:
            List of moving averages
        """
        if metric_name not in self.metrics:
            return []
        
        values = self.metrics[metric_name]
        moving_avg = []
        
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(values[start_idx:i+1]))
        
        return moving_avg
    
    def compute_win_rate(self, window: int = 100) -> Dict[str, float]:
        """
        Compute win rates with updated rules:
        - Predator wins ONLY if it caught prey (predator_reward > 0)
        - Prey wins if predator didn't catch it (predator_reward <= 0) OR if tied
        
        Args:
            window: Window size
        
        Returns:
            Dictionary with win rates (no draws - prey wins ties)
        """
        recent_data = self.episode_data[-window:]
        
        if len(recent_data) == 0:
            return {'predator_win_rate': 0.0, 'prey_win_rate': 0.0, 'draw_rate': 0.0}
        
        # Predator wins ONLY if it has positive reward (actually caught prey)
        predator_wins = sum(1 for d in recent_data if d['predator_reward'] > 0)
        
        # Prey wins if predator didn't catch it (predator_reward <= 0) OR if equal rewards
        prey_wins = len(recent_data) - predator_wins
        
        return {
            'predator_win_rate': predator_wins / len(recent_data),
            'prey_win_rate': prey_wins / len(recent_data),
            'draw_rate': 0.0  # No draws - ties go to prey
        }
    
    def save(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'metrics': {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for v in values] 
                       for k, values in self.metrics.items()},
            'episode_data': self.episode_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = defaultdict(list, data['metrics'])
        self.episode_data = data['episode_data']
    
    def print_summary(self):
        """Print summary of training metrics."""
        if len(self.episode_data) == 0:
            print("No data to summarize.")
            return
        
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Total Episodes: {len(self.episode_data)}")
        print(f"\nFinal 100 Episodes:")
        
        recent_stats = self.get_recent_stats(100)
        print(f"  Avg Predator Reward: {recent_stats['avg_predator_reward']:.2f} "
              f"(± {recent_stats['std_predator_reward']:.2f})")
        print(f"  Avg Prey Reward: {recent_stats['avg_prey_reward']:.2f} "
              f"(± {recent_stats['std_prey_reward']:.2f})")
        print(f"  Avg Episode Length: {recent_stats['avg_episode_length']:.2f}")
        
        win_rates = self.compute_win_rate(100)
        print(f"\nWin Rates (last 100 episodes):")
        print(f"  Predators: {win_rates['predator_win_rate']*100:.1f}%")
        print(f"  Prey: {win_rates['prey_win_rate']*100:.1f}%")
        print(f"  Draws: {win_rates['draw_rate']*100:.1f}%")
        print("="*60 + "\n")

