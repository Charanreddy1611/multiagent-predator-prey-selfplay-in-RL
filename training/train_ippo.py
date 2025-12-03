"""
Training script for IPPO (Independent PPO) algorithm.
Uses PettingZoo's Simple Tag environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from envs.pettingzoo_prey_predator import SimpleTagEnv
from algorithms.ippo import IPPO
from self_play.population import PopulationSelfPlay
from analysis.metrics import MetricsTracker
import argparse


def train_ippo(
    n_episodes: int = 5000,
    n_adversaries: int = 3,
    n_good: int = 1,
    max_cycles: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints"
):
    """
    Train adversaries and good agents using IPPO with population-based self-play.
    Uses PettingZoo's Simple Tag environment.
    
    Args:
        n_episodes: Number of training episodes
        n_adversaries: Number of adversaries (predators)
        n_good: Number of good agents (prey)
        max_cycles: Maximum cycles per episode
        device: Device for training
        save_dir: Directory to save checkpoints
    """
    print("="*60)
    print("Training IPPO with Population-Based Self-Play")
    print("PettingZoo Simple Tag Environment")
    print("="*60)
    print(f"Device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"Adversaries (Predators): {n_adversaries}")
    print(f"Good Agents (Prey): {n_good}")
    print(f"Max Cycles: {max_cycles}")
    print("="*60 + "\n")
    
    # Create Simple Tag environment
    env = SimpleTagEnv(
        n_adversaries=n_adversaries,
        n_good=n_good,
        max_cycles=max_cycles,
        render_mode=None
    )
    
    # Get observation and action dimensions
    sample_adversary = env.adversaries[0]
    sample_good = env.good_agents[0]
    
    obs_dim_adversary = env.observation_space(sample_adversary).shape[0]
    action_dim_adversary = env.action_space(sample_adversary).n
    
    obs_dim_good = env.observation_space(sample_good).shape[0]
    action_dim_good = env.action_space(sample_good).n
    
    # Create IPPO agents for adversaries (predators) and good agents (prey)
    predator_agent = IPPO(
        obs_dim=obs_dim_adversary,
        action_dim=action_dim_adversary,
        n_agents=n_adversaries,
        device=device
    )
    
    prey_agent = IPPO(
        obs_dim=obs_dim_good,
        action_dim=action_dim_good,
        n_agents=n_good,
        device=device
    )
    
    # Create population-based self-play wrapper
    self_play = PopulationSelfPlay(
        predator_agent=predator_agent,
        prey_agent=prey_agent,
        population_size=5,
        update_interval=500,
        sampling_strategy="fitness"
    )
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
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
        
        # Update agents
        update_metrics = self_play.update()
        
        # Track metrics
        metrics.log_episode(
            episode=episode,
            predator_reward=episode_reward_predators / n_adversaries,
            prey_reward=episode_reward_prey / n_good,
            episode_length=step_count
        )
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_stats = metrics.get_recent_stats(100)
            pop_stats = self_play.get_stats()
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Predator Reward: {avg_stats['avg_predator_reward']:.2f}")
            print(f"  Avg Prey Reward: {avg_stats['avg_prey_reward']:.2f}")
            print(f"  Avg Episode Length: {avg_stats['avg_episode_length']:.1f}")
            print(f"  Population Sizes: P={pop_stats['predator_population_size']}, "
                  f"Pr={pop_stats['prey_population_size']}")
            print()
        
        # Save checkpoints
        if (episode + 1) % 1000 == 0:
            predator_agent.save(f"{save_dir}/ippo_predator_ep{episode+1}.pt")
            prey_agent.save(f"{save_dir}/ippo_prey_ep{episode+1}.pt")
            metrics.save(f"{save_dir}/metrics_ep{episode+1}.json")
            print(f"✓ Checkpoint saved at episode {episode+1}\n")
    
    # Save final models
    predator_agent.save(f"{save_dir}/ippo_predator_final.pt")
    prey_agent.save(f"{save_dir}/ippo_prey_final.pt")
    metrics.save(f"{save_dir}/metrics_final.json")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPPO agents on Simple Tag")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--adversaries", type=int, default=3, help="Number of adversaries (predators)")
    parser.add_argument("--good", type=int, default=1, help="Number of good agents (prey)")
    parser.add_argument("--max-cycles", type=int, default=200, help="Maximum cycles per episode")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    train_ippo(
        n_episodes=args.episodes,
        n_adversaries=args.adversaries,
        n_good=args.good,
        max_cycles=args.max_cycles,
        save_dir=args.save_dir
    )

