"""
Optimized hyperparameter configurations for MARL algorithms.
Fine-tuned for PettingZoo Simple Tag environment.
"""

# MAPPO Hyperparameters - Optimized for predator-prey scenario
MAPPO_CONFIG = {
    # Network Architecture
    "hidden_dim": 256,  # Increased from 128 for better capacity
    
    # Learning Parameters
    "lr": 5e-4,  # Increased from 1e-4 for faster learning
    "gamma": 0.98,  # Slightly reduced for better short-term rewards
    
    # PPO-specific
    "clip_param": 0.3,  # Increased from 0.2 for more aggressive updates
    "n_epochs": 10,  # Increased from 4 for better policy optimization
    "batch_size": 256,  # Mini-batch size for updates
    
    # Loss coefficients
    "value_coef": 1.0,  # Increased from 0.5 for better value estimation
    "entropy_coef": 0.05,  # Increased from 0.01 for more exploration
    
    # Gradient clipping
    "max_grad_norm": 1.0,  # Increased from 0.5 for less aggressive clipping
    
    # Training
    "update_frequency": 25,  # Update every N steps (not just end of episode)
}

# IPPO Hyperparameters - Optimized for independent learning
IPPO_CONFIG = {
    # Network Architecture
    "hidden_dim": 256,  # Increased from 128
    
    # Learning Parameters
    "lr": 5e-4,  # Kept higher for faster convergence
    "gamma": 0.98,  # Slightly reduced
    
    # PPO-specific
    "clip_param": 0.3,  # Increased from 0.2
    "n_epochs": 10,  # Increased from 4
    "batch_size": 256,
    
    # Loss coefficients
    "value_coef": 1.0,  # Increased from 0.5
    "entropy_coef": 0.05,  # Increased from 0.01 for exploration
    
    # Gradient clipping
    "max_grad_norm": 1.0,  # Increased from 0.5
    
    # Training
    "update_frequency": 25,
}

# Environment-specific configurations
ENV_CONFIG = {
    "max_cycles": 200,
    "n_adversaries": 3,
    "n_good": 1,
    
    # Reward shaping (optional - can be added to environment wrapper)
    "survival_bonus": 0.1,  # Small reward for prey staying alive
    "capture_penalty": -10,  # Penalty when prey is caught
    "tag_reward": 10,  # Reward for predators when tagging
}

# Training configurations
TRAINING_CONFIG = {
    "n_episodes": 10000,  # Extended from 5000
    "warmup_episodes": 100,  # Episodes before starting updates
    "log_interval": 50,  # Log every N episodes
    "save_interval": 500,  # Save checkpoint every N episodes
    "eval_interval": 200,  # Evaluate without exploration every N episodes
    "eval_episodes": 10,  # Number of evaluation episodes
}

# Self-play configurations
SELFPLAY_CONFIG = {
    "alternating": {
        "switch_interval": 500,  # Reduced from 1000 for more frequent updates
    },
    "population": {
        "population_size": 5,
        "update_interval": 250,  # Reduced from 500
        "sampling_strategy": "fitness",
    },
    "league": {
        "main_agents": 2,
        "exploiter_agents": 2,
        "update_interval": 300,
    }
}

# Curriculum learning (optional advanced feature)
CURRICULUM_CONFIG = {
    "enabled": True,
    "stages": [
        # Stage 1: Easy - fewer predators, more prey
        {"episodes": 2000, "n_adversaries": 2, "n_good": 2, "max_cycles": 150},
        # Stage 2: Medium - balanced
        {"episodes": 3000, "n_adversaries": 3, "n_good": 1, "max_cycles": 200},
        # Stage 3: Hard - more predators
        {"episodes": 5000, "n_adversaries": 4, "n_good": 1, "max_cycles": 250},
    ]
}

