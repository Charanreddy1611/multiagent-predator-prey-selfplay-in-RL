"""
Evaluation script for trained multi-agent models.
Uses PettingZoo's Simple Tag environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from envs.pettingzoo_prey_predator import SimpleTagEnv
from algorithms.mappo import MAPPO
from algorithms.ippo import IPPO
import argparse
import time


def evaluate(
    algorithm: str = "mappo",
    predator_checkpoint: str = None,
    prey_checkpoint: str = None,
    n_episodes: int = 100,
    n_adversaries: int = 3,
    n_good: int = 1,
    max_cycles: int = 200,
    render: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate trained agents on Simple Tag.
    
    Args:
        algorithm: Algorithm type ("mappo" or "ippo")
        predator_checkpoint: Path to adversary checkpoint
        prey_checkpoint: Path to good agent checkpoint
        n_episodes: Number of evaluation episodes
        n_adversaries: Number of adversaries (predators)
        n_good: Number of good agents (prey)
        max_cycles: Maximum cycles per episode
        render: Whether to render episodes
        device: Device for evaluation
    """
    print("="*60)
    print(f"Evaluating {algorithm.upper()} Agents")
    print("PettingZoo Simple Tag Environment")
    print("="*60)
    print(f"Adversary Checkpoint: {predator_checkpoint}")
    print(f"Good Agent Checkpoint: {prey_checkpoint}")
    print(f"Episodes: {n_episodes}")
    print("="*60 + "\n")
    
    # Create Simple Tag environment
    env = SimpleTagEnv(
        n_adversaries=n_adversaries,
        n_good=n_good,
        max_cycles=max_cycles,
        render_mode="human" if render else None
    )
    
    # Get dimensions
    sample_adversary = env.adversaries[0]
    sample_good = env.good_agents[0]
    
    obs_dim_adversary = env.observation_space(sample_adversary).shape[0]
    action_dim_adversary = env.action_space(sample_adversary).n
    
    obs_dim_good = env.observation_space(sample_good).shape[0]
    action_dim_good = env.action_space(sample_good).n
    
    # Create and load agents
    if algorithm.lower() == "mappo":
        predator_agent = MAPPO(obs_dim_adversary, action_dim_adversary, n_adversaries, device=device)
        prey_agent = MAPPO(obs_dim_good, action_dim_good, n_good, device=device)
    elif algorithm.lower() == "ippo":
        predator_agent = IPPO(obs_dim_adversary, action_dim_adversary, n_adversaries, device=device)
        prey_agent = IPPO(obs_dim_good, action_dim_good, n_good, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load checkpoints
    if predator_checkpoint:
        predator_agent.load(predator_checkpoint)
        print(f"✓ Loaded predator checkpoint")
    if prey_checkpoint:
        prey_agent.load(prey_checkpoint)
        print(f"✓ Loaded prey checkpoint")
    
    # Evaluation metrics
    total_predator_reward = 0
    total_prey_reward = 0
    total_tags = 0  # Count close proximity events
    total_survival_time = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward_predators = 0
        episode_reward_prey = 0
        tags_this_episode = 0  # For Simple Tag, we count close proximity
        done = False
        step_count = 0
        
        while not done:
            # Get observations
            predator_obs = [obs[agent] for agent in env.adversaries if agent in obs]
            prey_obs = [obs[agent] for agent in env.good_agents if agent in obs]
            
            # Select actions (deterministic for evaluation)
            predator_actions = predator_agent.select_actions(predator_obs, deterministic=True)
            prey_actions = prey_agent.select_actions(prey_obs, deterministic=True)
            
            # Combine actions
            env_actions = {}
            for i, agent in enumerate(env.adversaries):
                if agent in env.agents:
                    env_actions[agent] = predator_actions[i]
            for i, agent in enumerate(env.good_agents):
                if agent in env.agents:
                    env_actions[agent] = prey_actions[i]
            
            # Step
            next_obs, rewards, terminations, truncations, infos = env.step(env_actions)
            
            # Track metrics
            predator_rewards = [rewards.get(agent, 0) for agent in env.adversaries]
            prey_rewards = [rewards.get(agent, 0) for agent in env.good_agents]
            
            episode_reward_predators += sum(predator_rewards)
            episode_reward_prey += sum(prey_rewards)
            
            # Count "tags" (when adversaries get positive rewards from proximity)
            # In Simple Tag, adversaries get rewarded for being close to good agents
            for i, agent in enumerate(env.adversaries):
                if agent in rewards:
                    # If adversary gets a positive reward, it's close to prey
                    if rewards[agent] > 0:
                        tags_this_episode += 1
            
            if render:
                env.render()
                time.sleep(0.1)
            
            obs = next_obs
            step_count += 1
            done = (all(terminations.values()) if terminations else False) or \
                   (all(truncations.values()) if truncations else False) or \
                   len(env.agents) == 0
        
        total_predator_reward += episode_reward_predators
        total_prey_reward += episode_reward_prey
        total_tags += tags_this_episode
        total_survival_time += step_count
        episode_lengths.append(step_count)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Predator Reward: {episode_reward_predators:.2f}, "
                  f"Prey Reward: {episode_reward_prey:.2f}, "
                  f"Proximity Events: {tags_this_episode}, "
                  f"Length: {step_count}")
    
    # Final statistics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Average Predator Reward: {total_predator_reward / n_episodes:.2f}")
    print(f"Average Prey Reward: {total_prey_reward / n_episodes:.2f}")
    print(f"Average Proximity Events: {total_tags / n_episodes:.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Predator Success Rate: {(total_predator_reward / n_episodes) / np.mean(episode_lengths):.4f}")
    print(f"Prey Evasion Score: {abs(total_prey_reward / n_episodes) / np.mean(episode_lengths):.4f} (lower is better)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained agents on Simple Tag")
    parser.add_argument("--algorithm", type=str, default="mappo", choices=["mappo", "ippo"],
                       help="Algorithm type")
    parser.add_argument("--predator-checkpoint", type=str, required=True,
                       help="Path to adversary checkpoint")
    parser.add_argument("--prey-checkpoint", type=str, required=True,
                       help="Path to good agent checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--adversaries", type=int, default=3,
                       help="Number of adversaries (predators)")
    parser.add_argument("--good", type=int, default=1,
                       help="Number of good agents (prey)")
    parser.add_argument("--max-cycles", type=int, default=200,
                       help="Maximum cycles per episode")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    
    args = parser.parse_args()
    
    evaluate(
        algorithm=args.algorithm,
        predator_checkpoint=args.predator_checkpoint,
        prey_checkpoint=args.prey_checkpoint,
        n_episodes=args.episodes,
        n_adversaries=args.adversaries,
        n_good=args.good,
        max_cycles=args.max_cycles,
        render=args.render
    )

