"""
Fair comparison between MAPPO and IPPO algorithms using League-Based Self-Play.
Both use the same:
- League-based self-play strategy (inspired by AlphaStar)
- Environment configuration (1v1 by default)
- Optimized hyperparameters
- Training episodes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
from envs.pettingzoo_prey_predator import SimpleTagEnv
from algorithms.mappo import MAPPO
from algorithms.ippo import IPPO
from self_play.league import LeagueSelfPlay
from analysis.metrics import MetricsTracker
import argparse
import json
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')


def train_algorithm(
    algorithm_name: str,
    algorithm_class,
    n_episodes: int,
    n_adversaries: int,
    n_good: int,
    max_cycles: int,
    device: str,
    save_dir: str,
    warmup_episodes: int = 100,
    n_main_agents: int = 2,
    n_exploiters: int = 1,
    checkpoint_interval: int = 500
):
    """
    Train a single algorithm with league-based self-play and return metrics.
    
    Args:
        algorithm_name: Name of algorithm (MAPPO or IPPO)
        algorithm_class: Algorithm class (MAPPO or IPPO)
        n_episodes: Number of training episodes
        n_adversaries: Number of predators
        n_good: Number of prey
        max_cycles: Max steps per episode
        device: Device for training
        save_dir: Directory to save results
        warmup_episodes: Warmup period
        n_main_agents: Number of main learner agents
        n_exploiters: Number of exploiter agents
        checkpoint_interval: Episodes between checkpoints
    
    Returns:
        MetricsTracker with training history
    """
    print("\n" + "="*70)
    print(f"Training {algorithm_name} with League-Based Self-Play")
    print("="*70)
    print(f"Configuration: {n_adversaries} Predator(s) vs {n_good} Prey")
    print(f"Episodes: {n_episodes}")
    print(f"Main Agents: {n_main_agents}")
    print(f"Exploiters: {n_exploiters}")
    print(f"Checkpoint Interval: {checkpoint_interval} episodes")
    print(f"Warmup: {warmup_episodes}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Create environment
    env = SimpleTagEnv(
        n_adversaries=n_adversaries,
        n_good=n_good,
        max_cycles=max_cycles,
        render_mode=None
    )
    
    # Get dimensions
    sample_adversary = env.adversaries[0]
    sample_good = env.good_agents[0]
    
    obs_dim_adversary = env.observation_space(sample_adversary).shape[0]
    action_dim_adversary = env.action_space(sample_adversary).n
    
    obs_dim_good = env.observation_space(sample_good).shape[0]
    action_dim_good = env.action_space(sample_good).n
    
    # Create agents with optimized hyperparameters (same for fair comparison)
    predator_agent = algorithm_class(
        obs_dim=obs_dim_adversary,
        action_dim=action_dim_adversary,
        n_agents=n_adversaries,
        device=device
    )
    
    prey_agent = algorithm_class(
        obs_dim=obs_dim_good,
        action_dim=action_dim_good,
        n_agents=n_good,
        device=device
    )
    
    # Learning rate schedulers (same as train_mappo_optimized.py)
    schedulers_predator = []
    schedulers_prey = []
    
    if hasattr(predator_agent, 'actor_optimizers'):
        for opt in predator_agent.actor_optimizers:
            schedulers_predator.append(CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5))
        schedulers_predator.append(CosineAnnealingLR(predator_agent.critic_optimizer, T_max=n_episodes, eta_min=1e-5))
    elif hasattr(predator_agent, 'optimizers'):
        for opt in predator_agent.optimizers:
            schedulers_predator.append(CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5))
    
    if hasattr(prey_agent, 'actor_optimizers'):
        for opt in prey_agent.actor_optimizers:
            schedulers_prey.append(CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5))
        schedulers_prey.append(CosineAnnealingLR(prey_agent.critic_optimizer, T_max=n_episodes, eta_min=1e-5))
    elif hasattr(prey_agent, 'optimizers'):
        for opt in prey_agent.optimizers:
            schedulers_prey.append(CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5))
    
    # Use League-Based self-play strategy
    self_play = LeagueSelfPlay(
        predator_agent=predator_agent,
        prey_agent=prey_agent,
        n_main_agents=n_main_agents,
        n_exploiters=n_exploiters,
        historical_window=10,
        checkpoint_interval=checkpoint_interval,
        matchmaking_strategy="elo"  # Use ELO-based matchmaking
    )
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best prey reward
    best_prey_reward = -float('inf')
    
    # Training loop
    print(f"Starting {algorithm_name} training with League-Based Self-Play...")
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward_predators = 0
        episode_reward_prey = 0
        done = False
        step_count = 0
        
        while not done:
            # Organize observations
            predator_obs = [obs[agent] for agent in env.adversaries if agent in obs]
            prey_obs = [obs[agent] for agent in env.good_agents if agent in obs]
            
            observations = {
                'predators': predator_obs,
                'prey': prey_obs
            }
            
            # Select actions
            actions_dict = self_play.step(observations)
            
            # Combine actions for environment
            env_actions = {}
            for i, agent in enumerate(env.adversaries):
                if agent in env.agents:
                    env_actions[agent] = actions_dict['predators'][i]
            for i, agent in enumerate(env.good_agents):
                if agent in env.agents:
                    env_actions[agent] = actions_dict['prey'][i]
            
            # Environment step
            next_obs, rewards, terminations, truncations, _ = env.step(env_actions)
            
            # Extract rewards and dones
            predator_rewards = [rewards.get(agent, 0) for agent in env.adversaries]
            prey_rewards = [rewards.get(agent, 0) for agent in env.good_agents]
            
            predator_dones = [terminations.get(agent, False) for agent in env.adversaries]
            prey_dones = [terminations.get(agent, False) for agent in env.good_agents]
            
            # Organize next observations
            next_predator_obs = [next_obs.get(agent, predator_obs[i]) 
                                for i, agent in enumerate(env.adversaries)]
            next_prey_obs = [next_obs.get(agent, prey_obs[i]) 
                            for i, agent in enumerate(env.good_agents)]
            
            # Store transition
            self_play.store_transition(
                obs={'predators': predator_obs, 'prey': prey_obs},
                actions=actions_dict,
                rewards={'predators': predator_rewards, 'prey': prey_rewards},
                next_obs={'predators': next_predator_obs, 'prey': next_prey_obs},
                dones={'predators': predator_dones, 'prey': prey_dones}
            )
            
            episode_reward_predators += sum(predator_rewards)
            episode_reward_prey += sum(prey_rewards)
            
            obs = next_obs
            step_count += 1
            
            # Check if episode is done
            done = (all(terminations.values()) if terminations else False) or \
                   (all(truncations.values()) if truncations else False) or \
                   len(env.agents) == 0
        
        # Update agents (only after warmup)
        if episode >= warmup_episodes:
            self_play.update()
            
            # Step learning rate schedulers (after optimizer.step)
            for scheduler in schedulers_predator:
                scheduler.step()
            for scheduler in schedulers_prey:
                scheduler.step()
        
        # Track metrics
        avg_pred_reward = episode_reward_predators / n_adversaries
        avg_prey_reward = episode_reward_prey / n_good
        
        metrics.log_episode(
            episode=episode,
            predator_reward=avg_pred_reward,
            prey_reward=avg_prey_reward,
            episode_length=step_count
        )
        
        # Update best prey reward
        if avg_prey_reward > best_prey_reward:
            best_prey_reward = avg_prey_reward
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_stats = metrics.get_recent_stats(100)
            league_stats = self_play.get_league_stats()
            print(f"[{algorithm_name}] Episode {episode + 1}/{n_episodes}")
            print(f"  Predator Reward: {avg_stats['avg_predator_reward']:7.2f} (std: {avg_stats['std_predator_reward']:.2f})")
            print(f"  Prey Reward:     {avg_stats['avg_prey_reward']:7.2f} (std: {avg_stats['std_prey_reward']:.2f})")
            print(f"  Best Prey:       {best_prey_reward:7.2f}")
            print(f"  League Sizes: Pred={league_stats['predator_league_size']}, Prey={league_stats['prey_league_size']}")
            
            # Show current learning rate
            if schedulers_prey and len(schedulers_prey) > 0:
                current_lr = schedulers_prey[0].get_last_lr()[0]
                print(f"  Current LR:      {current_lr:.6f}")
            print()
    
    # Save results
    predator_agent.save(f"{save_dir}/{algorithm_name.lower()}_predator_final.pt")
    prey_agent.save(f"{save_dir}/{algorithm_name.lower()}_prey_final.pt")
    metrics.save(f"{save_dir}/{algorithm_name.lower()}_metrics.json")
    
    league_stats = self_play.get_league_stats()
    print(f"\n✓ {algorithm_name} training complete!")
    print(f"  Best Prey Reward: {best_prey_reward:.2f}")
    print(f"  Final League Sizes: Pred={league_stats['predator_league_size']}, Prey={league_stats['prey_league_size']}")
    print(f"  Models saved to {save_dir}/")
    
    return metrics


def plot_comparison(mappo_metrics, ippo_metrics, save_path):
    """Create comparison plots between MAPPO and IPPO."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MAPPO vs IPPO Comparison (League-Based Self-Play)', 
                 fontsize=16, fontweight='bold')
    
    # Get data
    mappo_pred = mappo_metrics.get_all_metrics()['predator_rewards']
    mappo_prey = mappo_metrics.get_all_metrics()['prey_rewards']
    ippo_pred = ippo_metrics.get_all_metrics()['predator_rewards']
    ippo_prey = ippo_metrics.get_all_metrics()['prey_rewards']
    
    # Compute moving averages
    window = 100
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    mappo_pred_ma = moving_average(mappo_pred, window)
    mappo_prey_ma = moving_average(mappo_prey, window)
    ippo_pred_ma = moving_average(ippo_pred, window)
    ippo_prey_ma = moving_average(ippo_prey, window)
    
    # Plot 1: Predator Rewards
    ax1 = axes[0, 0]
    ax1.plot(mappo_pred_ma, label='MAPPO', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(ippo_pred_ma, label='IPPO', color='green', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Predator Rewards (Moving Avg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prey Rewards (MOST IMPORTANT)
    ax2 = axes[0, 1]
    ax2.plot(mappo_prey_ma, label='MAPPO', color='red', linewidth=2, alpha=0.7)
    ax2.plot(ippo_prey_ma, label='IPPO', color='orange', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Prey Rewards (Moving Avg) - KEY METRIC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    ax3 = axes[1, 0]
    mappo_lengths = mappo_metrics.get_all_metrics()['episode_lengths']
    ippo_lengths = ippo_metrics.get_all_metrics()['episode_lengths']
    mappo_len_ma = moving_average(mappo_lengths, window)
    ippo_len_ma = moving_average(ippo_lengths, window)
    ax3.plot(mappo_len_ma, label='MAPPO', color='purple', linewidth=2, alpha=0.7)
    ax3.plot(ippo_len_ma, label='IPPO', color='brown', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length (Moving Avg)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Performance Comparison (Bar Chart)
    ax4 = axes[1, 1]
    
    # Last 500 episodes statistics
    mappo_final_prey = np.mean(mappo_prey[-500:])
    ippo_final_prey = np.mean(ippo_prey[-500:])
    mappo_final_pred = np.mean(mappo_pred[-500:])
    ippo_final_pred = np.mean(ippo_pred[-500:])
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, [mappo_final_pred, mappo_final_prey], width, 
                    label='MAPPO', color=['blue', 'red'], alpha=0.7)
    bars2 = ax4.bar(x + width/2, [ippo_final_pred, ippo_final_prey], width, 
                    label='IPPO', color=['green', 'orange'], alpha=0.7)
    
    ax4.set_ylabel('Average Reward (Last 500 Episodes)')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Predator', 'Prey'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.show()


def print_comparison_summary(mappo_metrics, ippo_metrics):
    """Print detailed comparison summary."""
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: MAPPO vs IPPO (League-Based Self-Play)")
    print("="*80)
    
    # Get data
    mappo_pred = mappo_metrics.get_all_metrics()['predator_rewards']
    mappo_prey = mappo_metrics.get_all_metrics()['prey_rewards']
    ippo_pred = ippo_metrics.get_all_metrics()['predator_rewards']
    ippo_prey = ippo_metrics.get_all_metrics()['prey_rewards']
    
    # Compute statistics for last 500 episodes
    mappo_stats = {
        'pred_mean': np.mean(mappo_pred[-500:]),
        'pred_std': np.std(mappo_pred[-500:]),
        'prey_mean': np.mean(mappo_prey[-500:]),
        'prey_std': np.std(mappo_prey[-500:]),
        'prey_best': np.max(mappo_prey),
    }
    
    ippo_stats = {
        'pred_mean': np.mean(ippo_pred[-500:]),
        'pred_std': np.std(ippo_pred[-500:]),
        'prey_mean': np.mean(ippo_prey[-500:]),
        'prey_std': np.std(ippo_prey[-500:]),
        'prey_best': np.max(ippo_prey),
    }
    
    print("\n📊 Final Performance (Last 500 Episodes):")
    print("-" * 80)
    
    print("\nMAPPO:")
    print(f"  Predator Reward: {mappo_stats['pred_mean']:7.2f} ± {mappo_stats['pred_std']:6.2f}")
    print(f"  Prey Reward:     {mappo_stats['prey_mean']:7.2f} ± {mappo_stats['prey_std']:6.2f}")
    print(f"  Best Prey:       {mappo_stats['prey_best']:7.2f}")
    
    print("\nIPPO:")
    print(f"  Predator Reward: {ippo_stats['pred_mean']:7.2f} ± {ippo_stats['pred_std']:6.2f}")
    print(f"  Prey Reward:     {ippo_stats['prey_mean']:7.2f} ± {ippo_stats['prey_std']:6.2f}")
    print(f"  Best Prey:       {ippo_stats['prey_best']:7.2f}")
    
    # Compute differences
    print("\n📈 Difference (MAPPO - IPPO):")
    print("-" * 80)
    
    pred_diff = mappo_stats['pred_mean'] - ippo_stats['pred_mean']
    prey_diff = mappo_stats['prey_mean'] - ippo_stats['prey_mean']
    
    print(f"  Predator Reward: {pred_diff:+7.2f}")
    print(f"  Prey Reward:     {prey_diff:+7.2f}")
    
    # Determine winner
    print("\n🏆 Winner:")
    print("-" * 80)
    
    if abs(prey_diff) < 5:
        print("  DRAW - Both algorithms perform similarly")
    elif prey_diff > 0:
        print(f"  MAPPO wins by {prey_diff:.2f} points in prey reward")
        print("  → MAPPO's centralized critic helps with league complexity")
    else:
        print(f"  IPPO wins by {abs(prey_diff):.2f} points in prey reward")
        print("  → IPPO's independent learning handles league diversity better")
    
    # Win rates
    mappo_win_rates = mappo_metrics.compute_win_rate(500)
    ippo_win_rates = ippo_metrics.compute_win_rate(500)
    
    print("\n🎯 Win Rates (Last 500 Episodes):")
    print("-" * 80)
    print("\nMAPPO:")
    print(f"  Predator Wins: {mappo_win_rates['predator_win_rate']*100:5.1f}%")
    print(f"  Prey Wins:     {mappo_win_rates['prey_win_rate']*100:5.1f}%")
    print(f"  Draws:         {mappo_win_rates['draw_rate']*100:5.1f}%")
    
    print("\nIPPO:")
    print(f"  Predator Wins: {ippo_win_rates['predator_win_rate']*100:5.1f}%")
    print(f"  Prey Wins:     {ippo_win_rates['prey_win_rate']*100:5.1f}%")
    print(f"  Draws:         {ippo_win_rates['draw_rate']*100:5.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare MAPPO vs IPPO with League-Based Self-Play")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--adversaries", type=int, default=2, help="Number of predators (default: 2 for fair MAPPO comparison)")
    parser.add_argument("--good", type=int, default=2, help="Number of prey (default: 2 for fair MAPPO comparison)")
    parser.add_argument("--max-cycles", type=int, default=200, help="Max cycles per episode")
    parser.add_argument("--save-dir", type=str, default="comparison_league", help="Save directory")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup episodes")
    parser.add_argument("--main-agents", type=int, default=2, help="Number of main agents")
    parser.add_argument("--exploiters", type=int, default=1, help="Number of exploiter agents")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Warning for 1v1 scenarios
    if args.adversaries == 1 and args.good == 1:
        print("\n" + "⚠️ "*40)
        print("WARNING: 1v1 scenarios heavily favor IPPO over MAPPO!")
        print("MAPPO needs multi-agent scenarios to show its coordination advantages.")
        print("Recommended: Use --adversaries 2 --good 2 for fair comparison")
        print("⚠️ "*40 + "\n")
        input("Press Enter to continue or Ctrl+C to cancel...")
    
    print("\n" + "="*80)
    print("FAIR COMPARISON: MAPPO vs IPPO (League-Based Self-Play)")
    print("="*80)
    print(f"Configuration: {args.adversaries} Predator(s) vs {args.good} Prey")
    print(f"Episodes: {args.episodes}")
    print(f"Self-Play Strategy: League-Based (Same for both)")
    print(f"Main Agents: {args.main_agents}")
    print(f"Exploiters: {args.exploiters}")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    print(f"Hyperparameters: Optimized (Same for both)")
    print(f"Device: {args.device}")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    
    # Train MAPPO
    print("\n" + "🔵 " + "="*75)
    print("TRAINING MAPPO WITH LEAGUE-BASED SELF-PLAY")
    print("="*78)
    mappo_metrics = train_algorithm(
        algorithm_name="MAPPO",
        algorithm_class=MAPPO,
        n_episodes=args.episodes,
        n_adversaries=args.adversaries,
        n_good=args.good,
        max_cycles=args.max_cycles,
        device=args.device,
        save_dir=save_dir,
        warmup_episodes=args.warmup,
        n_main_agents=args.main_agents,
        n_exploiters=args.exploiters,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Train IPPO
    print("\n" + "🟢 " + "="*75)
    print("TRAINING IPPO WITH LEAGUE-BASED SELF-PLAY")
    print("="*78)
    ippo_metrics = train_algorithm(
        algorithm_name="IPPO",
        algorithm_class=IPPO,
        n_episodes=args.episodes,
        n_adversaries=args.adversaries,
        n_good=args.good,
        max_cycles=args.max_cycles,
        device=args.device,
        save_dir=save_dir,
        warmup_episodes=args.warmup,
        n_main_agents=args.main_agents,
        n_exploiters=args.exploiters,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Compare results
    print_comparison_summary(mappo_metrics, ippo_metrics)
    
    # Create plots
    plot_path = f"{save_dir}/comparison_plot_league.png"
    plot_comparison(mappo_metrics, ippo_metrics, plot_path)
    
    # Save comparison data
    comparison_data = {
        'configuration': {
            'adversaries': args.adversaries,
            'good': args.good,
            'episodes': args.episodes,
            'max_cycles': args.max_cycles,
            'self_play': 'league-based',
            'main_agents': args.main_agents,
            'exploiters': args.exploiters,
            'checkpoint_interval': args.checkpoint_interval,
        },
        'mappo': {
            'final_pred_reward': float(np.mean(mappo_metrics.get_all_metrics()['predator_rewards'][-500:])),
            'final_prey_reward': float(np.mean(mappo_metrics.get_all_metrics()['prey_rewards'][-500:])),
            'best_prey_reward': float(np.max(mappo_metrics.get_all_metrics()['prey_rewards'])),
        },
        'ippo': {
            'final_pred_reward': float(np.mean(ippo_metrics.get_all_metrics()['predator_rewards'][-500:])),
            'final_prey_reward': float(np.mean(ippo_metrics.get_all_metrics()['prey_rewards'][-500:])),
            'best_prey_reward': float(np.max(ippo_metrics.get_all_metrics()['prey_rewards'])),
        }
    }
    
    with open(f"{save_dir}/comparison_summary_league.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ All results saved to {save_dir}/")
    print(f"✓ Comparison plot: {plot_path}")
    print(f"✓ Summary: {save_dir}/comparison_summary_league.json")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

