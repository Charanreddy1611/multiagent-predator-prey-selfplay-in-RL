"""
Visualization tools for multi-agent learning.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns


def plot_training_curves(
    metrics: Dict[str, List],
    save_path: Optional[str] = None,
    window: int = 100
):
    """
    Plot training curves for predator and prey agents.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save figure
        window: Window size for moving average
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rewards over time
    ax = axes[0, 0]
    if 'predator_rewards' in metrics:
        predator_ma = _moving_average(metrics['predator_rewards'], window)
        ax.plot(predator_ma, label='Predator Reward', color='red', linewidth=2)
    if 'prey_rewards' in metrics:
        prey_ma = _moving_average(metrics['prey_rewards'], window)
        ax.plot(prey_ma, label='Prey Reward', color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    ax = axes[0, 1]
    if 'episode_lengths' in metrics:
        lengths_ma = _moving_average(metrics['episode_lengths'], window)
        ax.plot(lengths_ma, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reward difference (competitive balance)
    ax = axes[1, 0]
    if 'predator_rewards' in metrics and 'prey_rewards' in metrics:
        reward_diff = np.array(metrics['predator_rewards']) - np.array(metrics['prey_rewards'])
        diff_ma = _moving_average(reward_diff.tolist(), window)
        ax.plot(diff_ma, color='purple', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Difference (Predator - Prey)')
    ax.set_title('Competitive Balance')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution (recent episodes)
    ax = axes[1, 1]
    recent_window = 500
    if 'predator_rewards' in metrics and 'prey_rewards' in metrics:
        recent_predator = metrics['predator_rewards'][-recent_window:]
        recent_prey = metrics['prey_rewards'][-recent_window:]
        
        ax.hist(recent_predator, bins=30, alpha=0.6, label='Predator', color='red')
        ax.hist(recent_prey, bins=30, alpha=0.6, label='Prey', color='blue')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reward Distribution (Last {recent_window} Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_emergent_behaviors(
    episode_data: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot analysis of emergent behaviors.
    
    Args:
        episode_data: List of episode data dictionaries
        save_path: Path to save figure
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = [d['episode'] for d in episode_data]
    predator_rewards = [d['predator_reward'] for d in episode_data]
    prey_rewards = [d['prey_reward'] for d in episode_data]
    episode_lengths = [d['episode_length'] for d in episode_data]
    
    # Plot 1: Win rate over time
    ax = axes[0, 0]
    window = 100
    predator_wins = []
    for i in range(len(episode_data)):
        start = max(0, i - window + 1)
        wins = sum(1 for j in range(start, i+1) 
                  if predator_rewards[j] > prey_rewards[j])
        predator_wins.append(wins / min(i+1, window))
    
    ax.plot(episodes, predator_wins, color='red', linewidth=2, label='Predator Win Rate')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Balance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.set_title(f'Predator Win Rate (Window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Learning curve comparison
    ax = axes[0, 1]
    window = 100
    pred_ma = _moving_average(predator_rewards, window)
    prey_ma = _moving_average(prey_rewards, window)
    
    ax.plot(episodes, pred_ma, color='red', linewidth=2, label='Predator', alpha=0.8)
    ax.plot(episodes, prey_ma, color='blue', linewidth=2, label='Prey', alpha=0.8)
    ax.fill_between(episodes, pred_ma, prey_ma, 
                     where=np.array(pred_ma) >= np.array(prey_ma),
                     interpolate=True, alpha=0.3, color='red', label='Predator Advantage')
    ax.fill_between(episodes, pred_ma, prey_ma,
                     where=np.array(pred_ma) < np.array(prey_ma),
                     interpolate=True, alpha=0.3, color='blue', label='Prey Advantage')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Strategy adaptation (rolling correlation)
    ax = axes[1, 0]
    window = 100
    correlations = []
    for i in range(window, len(episode_data)):
        pred_window = predator_rewards[i-window:i]
        prey_window = prey_rewards[i-window:i]
        corr = np.corrcoef(pred_window, prey_window)[0, 1]
        correlations.append(corr)
    
    ax.plot(episodes[window:], correlations, color='purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Predator-Prey Reward Correlation (Adaptation)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    # Plot 4: Episode length vs performance
    ax = axes[1, 1]
    # Bin by episode length and show average rewards
    length_bins = np.linspace(min(episode_lengths), max(episode_lengths), 20)
    bin_indices = np.digitize(episode_lengths, length_bins)
    
    avg_pred_by_length = []
    avg_prey_by_length = []
    bin_centers = []
    
    for i in range(1, len(length_bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            avg_pred_by_length.append(np.mean([predator_rewards[j] for j in range(len(mask)) if mask[j]]))
            avg_prey_by_length.append(np.mean([prey_rewards[j] for j in range(len(mask)) if mask[j]]))
            bin_centers.append((length_bins[i-1] + length_bins[i]) / 2)
    
    ax.scatter(bin_centers, avg_pred_by_length, color='red', s=100, alpha=0.6, label='Predator')
    ax.scatter(bin_centers, avg_prey_by_length, color='blue', s=100, alpha=0.6, label='Prey')
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Average Reward')
    ax.set_title('Episode Length vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_heatmap(
    position_data: np.ndarray,
    title: str = "Agent Position Heatmap",
    save_path: Optional[str] = None
):
    """
    Plot heatmap of agent positions.
    
    Args:
        position_data: 2D array of position frequencies
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(position_data, cmap='YlOrRd', cbar=True, square=True)
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()


def _moving_average(data: List[float], window: int) -> List[float]:
    """Compute moving average."""
    if len(data) < window:
        window = len(data)
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(data[start_idx:i+1]))
    
    return moving_avg

