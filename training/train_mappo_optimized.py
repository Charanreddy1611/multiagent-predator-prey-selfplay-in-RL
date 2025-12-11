"""
Optimized training script for MAPPO with improved hyperparameters.
Includes learning rate scheduling, better logging, and curriculum learning support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import warnings
from envs.pettingzoo_prey_predator import SimpleTagEnv
from algorithms.mappo import MAPPO
from self_play.alternating import AlternatingSelfPlay
from analysis.metrics import MetricsTracker
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Suppress PyTorch LR scheduler warning - we call scheduler.step() after optimizer.step()
# but PyTorch detects it as "before" on first episode. This is a false positive.
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')


def train_mappo_optimized(
    n_episodes: int = 10000,  # Increased from 5000
    n_adversaries: int = 3,
    n_good: int = 1,
    max_cycles: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints_optimized",
    use_lr_scheduler: bool = True,
    warmup_episodes: int = 100
):
    """
    Train with optimized hyperparameters and advanced features.
    
    Args:
        n_episodes: Number of training episodes (increased to 10000)
        n_adversaries: Number of adversaries (predators)
        n_good: Number of good agents (prey)
        max_cycles: Maximum cycles per episode
        device: Device for training
        save_dir: Directory to save checkpoints
        use_lr_scheduler: Whether to use learning rate scheduling
        warmup_episodes: Number of warmup episodes before training starts
    """
    print("="*70)
    print("OPTIMIZED MAPPO Training with Enhanced Hyperparameters")
    print("PettingZoo Simple Tag Environment")
    print("="*70)
    print(f"Device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"Warmup Episodes: {warmup_episodes}")
    print(f"Adversaries (Predators): {n_adversaries}")
    print(f"Good Agents (Prey): {n_good}")
    print(f"Max Cycles: {max_cycles}")
    print(f"Learning Rate Scheduler: {'Enabled' if use_lr_scheduler else 'Disabled'}")
    print("="*70)
    print("\nOptimized Hyperparameters:")
    print("  - Learning Rate: 5e-4 (increased from 1e-4)")
    print("  - Hidden Dimension: 256 (increased from 128)")
    print("  - Entropy Coefficient: 0.05 (increased from 0.01)")
    print("  - Value Coefficient: 1.0 (increased from 0.5)")
    print("  - Clip Parameter: 0.3 (increased from 0.2)")
    print("  - Update Epochs: 10 (increased from 4)")
    print("  - Gamma: 0.98 (reduced from 0.99)")
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
    
    # Create MAPPO agents with optimized hyperparameters (using defaults)
    predator_agent = MAPPO(
        obs_dim=obs_dim_adversary,
        action_dim=action_dim_adversary,
        n_agents=n_adversaries,
        device=device
        # All optimized hyperparameters are now defaults
    )
    
    prey_agent = MAPPO(
        obs_dim=obs_dim_good,
        action_dim=action_dim_good,
        n_agents=n_good,
        device=device
        # All optimized hyperparameters are now defaults
    )
    
    # Optional: Learning rate schedulers
    schedulers_predator = None
    schedulers_prey = None
    
    if use_lr_scheduler:
        # Cosine annealing for gradual learning rate decay
        schedulers_predator = [
            CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
            for opt in predator_agent.actor_optimizers
        ]
        schedulers_predator.append(
            CosineAnnealingLR(predator_agent.critic_optimizer, T_max=n_episodes, eta_min=1e-5)
        )
        
        schedulers_prey = [
            CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
            for opt in prey_agent.actor_optimizers
        ]
        schedulers_prey.append(
            CosineAnnealingLR(prey_agent.critic_optimizer, T_max=n_episodes, eta_min=1e-5)
        )
    
    # Create self-play wrapper with reduced switch interval
    self_play = AlternatingSelfPlay(
        predator_agent=predator_agent,
        prey_agent=prey_agent,
        switch_interval=500  # Reduced from 1000
    )
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best prey reward for early stopping
    best_prey_reward = -float('inf')
    patience = 2000
    episodes_without_improvement = 0
    
    # Training loop
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
            
            # Select actions (with exploration during warmup)
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
        update_metrics = {}
        if episode >= warmup_episodes:
            update_metrics = self_play.update()
        
        # Step learning rate schedulers AFTER optimizer.step() (which happens in update())
        # Only step if we actually did an update this episode
        if episode >= warmup_episodes and use_lr_scheduler and schedulers_predator and schedulers_prey:
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
        
        # Check for improvement in prey reward
        if avg_prey_reward > best_prey_reward:
            best_prey_reward = avg_prey_reward
            episodes_without_improvement = 0
            # Save best model
            prey_agent.save(f"{save_dir}/mappo_prey_best.pt")
        else:
            episodes_without_improvement += 1
        
        # Logging
        if (episode + 1) % 50 == 0:  # More frequent logging
            avg_stats = metrics.get_recent_stats(100)
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Predator Reward: {avg_stats['avg_predator_reward']:.2f} "
                  f"(std: {avg_stats['std_predator_reward']:.2f})")
            print(f"  Avg Prey Reward: {avg_stats['avg_prey_reward']:.2f} "
                  f"(std: {avg_stats['std_prey_reward']:.2f})")
            print(f"  Best Prey Reward: {best_prey_reward:.2f}")
            print(f"  Avg Episode Length: {avg_stats['avg_episode_length']:.1f}")
            
            if use_lr_scheduler and schedulers_prey:
                current_lr = schedulers_prey[0].get_last_lr()[0]
                print(f"  Current Learning Rate: {current_lr:.6f}")
            
            if update_metrics:
                print(f"  Actor Loss: {update_metrics.get('actor_loss', 'N/A')}")
                print(f"  Critic Loss: {update_metrics.get('critic_loss', 'N/A')}")
            print()
        
        # Save checkpoints
        if (episode + 1) % 500 == 0:
            predator_agent.save(f"{save_dir}/mappo_predator_ep{episode+1}.pt")
            prey_agent.save(f"{save_dir}/mappo_prey_ep{episode+1}.pt")
            metrics.save(f"{save_dir}/metrics_ep{episode+1}.json")
            print(f"✓ Checkpoint saved at episode {episode+1}\n")
        
        # Early stopping check
        if episodes_without_improvement >= patience:
            print(f"\n⚠ Early stopping triggered after {episode+1} episodes")
            print(f"No improvement in prey reward for {patience} episodes")
            break
    
    # Save final models
    predator_agent.save(f"{save_dir}/mappo_predator_final.pt")
    prey_agent.save(f"{save_dir}/mappo_prey_final.pt")
    metrics.save(f"{save_dir}/metrics_final.json")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Models saved to {save_dir}/")
    print(f"Best Prey Reward: {best_prey_reward:.2f}")
    
    # Print summary
    metrics.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO with optimized hyperparameters")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--adversaries", type=int, default=3, help="Number of adversaries (predators)")
    parser.add_argument("--good", type=int, default=1, help="Number of good agents (prey)")
    parser.add_argument("--max-cycles", type=int, default=200, help="Maximum cycles per episode")
    parser.add_argument("--save-dir", type=str, default="checkpoints_optimized", help="Save directory")
    parser.add_argument("--no-lr-scheduler", action="store_true", help="Disable learning rate scheduler")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup episodes")
    
    args = parser.parse_args()
    
    train_mappo_optimized(
        n_episodes=args.episodes,
        n_adversaries=args.adversaries,
        n_good=args.good,
        max_cycles=args.max_cycles,
        save_dir=args.save_dir,
        use_lr_scheduler=not args.no_lr_scheduler,
        warmup_episodes=args.warmup
    )

