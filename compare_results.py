"""
Compare training results between MAPPO and IPPO.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from analysis.plots import plot_training_curves

def load_metrics(filepath):
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def compare_algorithms():
    """Compare MAPPO and IPPO training results."""
    
    # Load metrics
    try:
        mappo_metrics = load_metrics('checkpoints/metrics_final.json')
        # Assuming IPPO metrics are also saved as metrics_final.json
        # You may need to rename one before running this
        print("Loaded metrics successfully!")
    except FileNotFoundError:
        print("Error: Could not find metrics files in checkpoints/")
        print("Make sure metrics_final.json exists")
        return
    
    # Extract data
    episodes = range(len(mappo_metrics['metrics']['predator_rewards']))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Predator Rewards
    ax = axes[0, 0]
    ax.plot(episodes, mappo_metrics['metrics']['predator_rewards'], 
            label='MAPPO', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Predator Reward')
    ax.set_title('Predator Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Prey Rewards
    ax = axes[0, 1]
    ax.plot(episodes, mappo_metrics['metrics']['prey_rewards'], 
            label='MAPPO', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Prey Reward')
    ax.set_title('Prey Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    ax = axes[1, 0]
    ax.plot(episodes, mappo_metrics['metrics']['episode_lengths'], 
            label='MAPPO', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Win Rate (Moving Average)
    ax = axes[1, 1]
    window = 100
    predator_rewards = np.array(mappo_metrics['metrics']['predator_rewards'])
    prey_rewards = np.array(mappo_metrics['metrics']['prey_rewards'])
    predator_wins = (predator_rewards > prey_rewards).astype(float)
    
    # Moving average
    win_rate = np.convolve(predator_wins, np.ones(window)/window, mode='valid')
    ax.plot(range(len(win_rate)), win_rate, label='MAPPO Predator Win Rate')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Balance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Predator Win Rate')
    ax.set_title(f'Win Rate (Moving Avg, window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to comparison_results.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)
    print(f"\nMAPPO:")
    print(f"  Final Predator Reward: {predator_rewards[-100:].mean():.2f}")
    print(f"  Final Prey Reward: {prey_rewards[-100:].mean():.2f}")
    print(f"  Final Episode Length: {np.array(mappo_metrics['metrics']['episode_lengths'][-100:]).mean():.2f}")
    print(f"  Predator Win Rate: {predator_wins[-100:].mean()*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    compare_algorithms()

