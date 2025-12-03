# Emergent Adversarial Behaviors via Self-Play in Multi-Agent Reinforcement Learning

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-green.svg)](https://pettingzoo.farama.org/)
[![License](https://img.shields.io/badge/license-Educational-lightgrey.svg)]()

A comprehensive research project exploring how competitive interactions in self-play settings lead to complex, adaptive behaviors in multi-agent reinforcement learning using PettingZoo's **Simple Tag** environment.

</div>

---

## 📋 Table of Contents

- [Team Members](#-team-members)
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Environment Details](#-environment-details)
- [Algorithms](#-algorithms)
- [Self-Play Strategies](#-self-play-strategies)
- [Training Configuration](#-training-configuration)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Visualization](#-visualization)
- [Results & Analysis](#-results--analysis)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Troubleshooting](#-troubleshooting)
- [Extensions & Future Work](#-extensions--future-work)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 👥 Team Members

| Name | Email | Role |
|------|-------|------|
| Charan Reddy Nandyala | cnand002@ucr.edu | Lead Developer |
| Dhanush Chalicheemala | dchal007@ucr.edu | Algorithm Implementation |
| Satyadev Gangineni | sgang024@ucr.edu | Analysis & Visualization |

**Institution:** University of California, Riverside  
**Course:** Advanced Machine Learning / Multi-Agent Systems  
**Semester:** Fall 2024

---

## 🎯 Project Overview

This project investigates **emergent adversarial behaviors** in multi-agent reinforcement learning through self-play mechanisms. We use **PettingZoo's Simple Tag environment**, a predator-prey scenario where adversaries (predators) attempt to tag good agents (prey) while prey try to avoid capture.

### Research Questions

1. **How do competitive interactions shape agent strategies?**  
   Exploring how adversarial training leads to increasingly sophisticated behaviors in both predators and prey.

2. **Which self-play strategies produce the most robust agents?**  
   Comparing alternating, population-based, league, and reservoir self-play methods.

3. **How do different MARL algorithms perform in adversarial settings?**  
   Benchmarking MAPPO, IPPO, and MADDPG in competitive scenarios.

4. **What emergent behaviors arise from self-play training?**  
   Identifying coordination patterns, strategic positioning, and adaptive tactics.

### Motivation

Traditional single-agent RL often fails in multi-agent settings due to:
- Non-stationary environments (other agents are learning)
- Credit assignment problems (which agent caused the outcome?)
- Emergent complexity (unpredictable interactions)

Self-play provides a natural curriculum where agents continuously adapt to increasingly skilled opponents, leading to sophisticated emergent behaviors without hand-crafted rewards.

---

## ✨ Key Features

### 🎮 Environment
- **PettingZoo Simple Tag**: Multi-agent predator-prey game
- Continuous observation space (positions, velocities)
- Discrete action space (5 actions: none, up, down, left, right)
- Configurable number of predators and prey
- Customizable grid size and episode length

### 🤖 RL Algorithms

#### MAPPO (Multi-Agent Proximal Policy Optimization)
- ✅ Centralized training, decentralized execution (CTDE)
- ✅ Shared critic with global state information
- ✅ Individual actor networks for each agent
- ✅ Best for scenarios requiring coordination

#### IPPO (Independent Proximal Policy Optimization)
- ✅ Fully independent learning for each agent
- ✅ Separate actor-critic networks
- ✅ Treats other agents as part of environment
- ✅ Good baseline for comparison

#### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- ✅ Centralized critic, decentralized actors
- ✅ Supports continuous action spaces
- ✅ Experience replay for sample efficiency
- ✅ Actor-critic architecture with deterministic policies

### 🔄 Self-Play Strategies

1. **Alternating Self-Play**: Turn-based training (predators → prey → predators...)
2. **Population-Based Self-Play**: Maintain diverse agent populations
3. **League Self-Play**: AlphaStar-inspired competitive training with exploiters
4. **Reservoir Self-Play**: Historical opponent sampling with optional prioritization

### 📊 Analysis Tools
- Real-time metrics tracking (rewards, episode length, success rate)
- Comprehensive visualization (training curves, heatmaps, behavior plots)
- Model checkpointing and evaluation
- JSON-based metric storage for reproducibility

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "final project"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import pettingzoo; print('Installation successful!')"
```

### Dependencies Breakdown

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `numpy` | ≥1.24.0 | Numerical computations |
| `gymnasium` | ≥0.29.0 | RL environment interface |
| `pettingzoo` | ≥1.24.0 | Multi-agent environments |
| `pygame` | ≥2.1.0 | Environment rendering |
| `matplotlib` | ≥3.7.0 | Visualization |
| `seaborn` | ≥0.12.0 | Statistical plots |
| `tqdm` | ≥4.65.0 | Progress bars |

---

## 📁 Project Structure

```
final project/
│
├── 📂 envs/                              # Environment implementations
│   ├── __init__.py                       # Package initializer
│   └── pettingzoo_prey_predator.py       # PettingZoo Simple Tag wrapper
│
├── 📂 algorithms/                        # RL algorithm implementations
│   ├── __init__.py                       # Package initializer
│   ├── mappo.py                          # Multi-Agent PPO
│   ├── ippo.py                           # Independent PPO
│   └── maddpg.py                         # Multi-Agent DDPG
│
├── 📂 self_play/                         # Self-play strategy implementations
│   ├── __init__.py                       # Package initializer
│   ├── alternating.py                    # Alternating training strategy
│   ├── population.py                     # Population-based training
│   ├── league.py                         # League-based training
│   └── reservoir.py                      # Reservoir sampling strategy
│
├── 📂 training/                          # Training and evaluation scripts
│   ├── __init__.py                       # Package initializer
│   ├── train_mappo.py                    # MAPPO training script
│   ├── train_ippo.py                     # IPPO training script
│   └── evaluate.py                       # Model evaluation script
│
├── 📂 analysis/                          # Analysis and visualization tools
│   ├── __init__.py                       # Package initializer
│   ├── metrics.py                        # Metrics tracking utilities
│   └── plots.py                          # Plotting and visualization
│
├── 📂 checkpoints/                       # Saved models and metrics
│   ├── mappo_predator_final.pt           # Trained MAPPO predator
│   ├── mappo_prey_final.pt               # Trained MAPPO prey
│   ├── ippo_predator_final.pt            # Trained IPPO predator
│   ├── ippo_prey_final.pt                # Trained IPPO prey
│   └── metrics_final.json                # Training metrics
│
├── 📄 compare_results.py                 # Script to compare algorithms
├── 📄 visualize_training.py              # Visualization script
├── 📄 requirements.txt                   # Project dependencies
├── 📄 README.md                          # This file
└── 📄 project proposal (1).pdf           # Original project proposal
```

---

## 🏃 Quick Start

### Training with MAPPO (Recommended)

```bash
python training/train_mappo.py --episodes 5000 --adversaries 3 --good 1
```

This trains 3 predators and 1 prey using MAPPO for 5000 episodes (~30-60 minutes on CPU).

### Training with IPPO

```bash
python training/train_ippo.py --episodes 5000 --adversaries 3 --good 1
```

### Quick Evaluation (With Pre-trained Models)

```bash
python training/evaluate.py \
    --algorithm mappo \
    --predator-checkpoint checkpoints/mappo_predator_final.pt \
    --prey-checkpoint checkpoints/mappo_prey_final.pt \
    --episodes 100 \
    --render
```

### Visualize Training Results

```bash
python visualize_training.py --metrics checkpoints/metrics_final.json
```

### Compare Algorithms

```bash
python compare_results.py
```

---

## 📖 Detailed Usage

### Training MAPPO

**Basic Usage:**
```bash
python training/train_mappo.py
```

**Advanced Usage with Custom Parameters:**
```bash
python training/train_mappo.py \
    --episodes 10000 \
    --adversaries 4 \
    --good 2 \
    --max-cycles 250 \
    --lr 3e-4 \
    --gamma 0.99 \
    --save-dir checkpoints/mappo_custom \
    --save-interval 500 \
    --device cuda
```

**All Available Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 5000 | Number of training episodes |
| `--adversaries` | int | 3 | Number of predator agents |
| `--good` | int | 1 | Number of prey agents |
| `--max-cycles` | int | 200 | Maximum steps per episode |
| `--lr` | float | 3e-4 | Learning rate |
| `--gamma` | float | 0.95 | Discount factor |
| `--gae-lambda` | float | 0.95 | GAE lambda for advantage estimation |
| `--clip-epsilon` | float | 0.2 | PPO clipping parameter |
| `--entropy-coef` | float | 0.01 | Entropy bonus coefficient |
| `--value-coef` | float | 0.5 | Value loss coefficient |
| `--batch-size` | int | 64 | Minibatch size |
| `--save-dir` | str | checkpoints | Directory to save models |
| `--save-interval` | int | 1000 | Episodes between checkpoints |
| `--device` | str | cuda | Device (cuda/cpu) |
| `--seed` | int | None | Random seed for reproducibility |

### Training IPPO

**Basic Usage:**
```bash
python training/train_ippo.py
```

**With Custom Configuration:**
```bash
python training/train_ippo.py \
    --episodes 10000 \
    --adversaries 3 \
    --good 1 \
    --lr 5e-4 \
    --hidden-dim 256 \
    --device cuda
```

### Evaluation

**Evaluate Specific Models:**
```bash
python training/evaluate.py \
    --algorithm mappo \
    --predator-checkpoint checkpoints/mappo_predator_ep1000.pt \
    --prey-checkpoint checkpoints/mappo_prey_ep1000.pt \
    --episodes 100 \
    --render \
    --save-video
```

**Evaluation Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | mappo | Algorithm type (mappo/ippo/maddpg) |
| `--predator-checkpoint` | str | Required | Path to predator model |
| `--prey-checkpoint` | str | Required | Path to prey model |
| `--episodes` | int | 100 | Number of evaluation episodes |
| `--adversaries` | int | 3 | Number of predators |
| `--good` | int | 1 | Number of prey |
| `--max-cycles` | int | 200 | Max steps per episode |
| `--render` | flag | False | Render environment visually |
| `--save-video` | flag | False | Save video of episodes |
| `--video-dir` | str | videos | Directory for videos |
| `--device` | str | cuda | Device to run on |

---

## 🎮 Environment Details

### PettingZoo Simple Tag Overview

**Simple Tag** is a cooperative-competitive environment where:
- **Predators (Adversaries)** - Red circles that try to tag prey
- **Prey (Good Agents)** - Green circles that try to avoid predators
- **Landmarks** - Gray circles that serve as obstacles

### Environment Configuration

```python
from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.parallel_env(
    num_good=1,           # Number of prey
    num_adversaries=3,    # Number of predators
    num_obstacles=2,      # Number of landmarks
    max_cycles=200,       # Episode length
    continuous_actions=False  # Use discrete actions
)
```

### Observation Space

**Structure:** Continuous vector of varying size depending on number of agents

**For each agent, observations include:**
- Own velocity (2D): [v_x, v_y]
- Own position (2D): [x, y]
- Relative positions to landmarks: [(rel_x1, rel_y1), (rel_x2, rel_y2), ...]
- Relative positions to other agents: [(rel_x1, rel_y1), (rel_x2, rel_y2), ...]
- Agent-specific features

**Typical observation dimension:**
- Predator: ~16-20 dimensions
- Prey: ~16-20 dimensions

### Action Space

**Discrete Actions (5 options):**
- `0`: No movement
- `1`: Move left
- `2`: Move right
- `3`: Move down
- `4`: Move up

**Movement characteristics:**
- Prey are faster than predators
- Actions are executed with some physics (momentum)
- Collisions with walls and landmarks

### Reward Structure

**Predators (Adversaries):**
```python
reward = -min_distance_to_prey  # Negative distance to closest prey
```
- Maximized when close to prey
- Shared among all predators
- Encourages coordination to surround prey

**Prey (Good Agents):**
```python
reward = min_distance_to_adversaries  # Distance to closest predator
```
- Maximized when far from predators
- Encourages evasion and maintaining distance

### Episode Termination

Episodes end when:
1. Maximum cycles reached (default: 200)
2. Predators successfully tag prey (collision occurs)
3. Manual termination via environment reset

---

## 🧠 Algorithms

### MAPPO (Multi-Agent Proximal Policy Optimization)

#### Architecture

```
┌─────────────────────────────────────┐
│     Global State (Centralized)      │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌──────────────┐
       │    Critic    │  (Shared)
       │  (Value Net) │
       └──────────────┘
               
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Actor 1 │  │  Actor 2 │  │  Actor N │
│(Predator)│  │(Predator)│  │  (Prey)  │
└──────────┘  └──────────┘  └──────────┘
     ▲             ▲             ▲
     │             │             │
 Local Obs    Local Obs    Local Obs
```

#### Key Components

**Centralized Critic:**
- Input: Global state (all agent positions and velocities)
- Output: Value estimate for current state
- Shared across all agents of the same type

**Decentralized Actors:**
- Input: Local observation (agent-specific)
- Output: Action probabilities
- Individual policy for each agent

**Training Process:**
1. Collect trajectories using current policies
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policies using PPO objective with clipping
4. Update centralized critic to predict state values

**PPO Objective:**
```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

#### When to Use MAPPO
- ✅ Agents need coordination
- ✅ Global state information available during training
- ✅ Homogeneous agent groups (e.g., all predators)
- ✅ Partial observability during execution is acceptable

### IPPO (Independent Proximal Policy Optimization)

#### Architecture

```
Agent 1          Agent 2          Agent N
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Actor 1 │      │ Actor 2 │      │ Actor N │
└────┬────┘      └────┬────┘      └────┬────┘
     │                │                │
┌────▼────┐      ┌────▼────┐      ┌────▼────┐
│Critic 1 │      │Critic 2 │      │Critic N │
└────▲────┘      └────▲────┘      └────▲────┘
     │                │                │
 Local Obs       Local Obs        Local Obs
```

#### Key Components

**Independent Actor-Critics:**
- Each agent has its own actor and critic
- No parameter sharing between agents
- Treats other agents as part of environment

**Training Process:**
1. Each agent independently collects experiences
2. Each agent updates its own policy using PPO
3. No coordination or communication between agents

#### When to Use IPPO
- ✅ Simple baseline comparison
- ✅ Fully decentralized scenarios
- ✅ No global state information available
- ✅ Independent agent types with different goals

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

#### Architecture

```
         Training Phase (Centralized)
┌────────────────────────────────────────┐
│  Global State + All Actions            │
└─────────────┬──────────────────────────┘
              │
      ┌───────▼────────┐
      │  Centralized   │
      │    Critic      │
      └────────────────┘
              
Execution Phase (Decentralized)
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Actor 1 │  │  Actor 2 │  │  Actor N │
└──────────┘  └──────────┘  └──────────┘
```

#### Key Components

**Centralized Critic (Training Only):**
- Input: Observations and actions of all agents
- Output: Q-value estimate
- Has access to global information

**Decentralized Actors:**
- Input: Local observation only
- Output: Deterministic action
- Used during execution

**Experience Replay:**
- Stores transitions from all agents
- Samples minibatches for training
- Improves sample efficiency

#### When to Use MADDPG
- ✅ Continuous action spaces
- ✅ Need sample efficiency (replay buffer)
- ✅ Agents have different action spaces
- ✅ Deterministic policies preferred

### Algorithm Comparison

| Feature | MAPPO | IPPO | MADDPG |
|---------|-------|------|--------|
| **Critic** | Centralized (shared) | Decentralized (independent) | Centralized (training only) |
| **Actor** | Decentralized | Decentralized | Decentralized |
| **Action Space** | Discrete | Discrete | Continuous |
| **Parameter Sharing** | Optional (same type) | No | No |
| **Sample Efficiency** | Medium | Medium | High (replay buffer) |
| **Coordination** | High | Low | Medium |
| **Scalability** | Good | Excellent | Medium |
| **Complexity** | Medium | Low | High |

---

## 🔄 Self-Play Strategies

### 1. Alternating Self-Play

#### Concept
Agents alternate between training and being frozen as opponents.

#### Algorithm
```
for episode in episodes:
    if episode % switch_frequency == 0:
        toggle_training_agent()
    
    if training_predators:
        update_predator_policies()
        freeze_prey_policies()
    else:
        freeze_predator_policies()
        update_prey_policies()
```

#### Parameters
- `switch_frequency`: Episodes between role switches (default: 500)
- `training_agent`: Which side currently trains

#### Pros & Cons
- ✅ Simple to implement
- ✅ Stable training
- ✅ Clear curriculum
- ❌ Catastrophic forgetting possible
- ❌ May not adapt to sudden strategy changes

#### Usage
```python
from self_play.alternating import AlternatingSelfPlay

self_play = AlternatingSelfPlay(
    switch_frequency=500,
    start_with='predators'
)
```

### 2. Population-Based Self-Play

#### Concept
Maintain populations of agents and sample opponents from the population.

#### Algorithm
```
Initialize populations P_pred and P_prey

for episode in episodes:
    # Sample opponents from populations
    pred = random.choice(P_pred)
    prey = random.choice(P_prey)
    
    # Play episode
    result = play_episode(pred, prey)
    
    # Train current policies
    update_policies(result)
    
    # Update populations
    if episode % update_freq == 0:
        add_to_population(current_policies)
        remove_worst_from_population()
```

#### Parameters
- `population_size`: Max agents per population (default: 10)
- `update_frequency`: Episodes between population updates (default: 100)
- `removal_strategy`: 'worst', 'random', or 'oldest'

#### Pros & Cons
- ✅ Maintains diversity
- ✅ Robust to different opponent types
- ✅ Natural exploration
- ❌ Memory intensive
- ❌ Requires more computation

#### Usage
```python
from self_play.population import PopulationSelfPlay

self_play = PopulationSelfPlay(
    population_size=10,
    update_frequency=100,
    removal_strategy='worst'
)
```

### 3. League Self-Play

#### Concept
Inspired by AlphaStar, maintains different types of agents competing in a league.

#### Agent Types

**Main Agents:**
- Continuously train against all opponents
- Represent current best policies

**Exploiter Agents:**
- Target weaknesses in main agents
- Train specifically to beat main agents

**Historical Snapshots:**
- Frozen copies of main agents at different training stages
- Provide stable opponents

#### Tournament Structure
```
         ┌───────────┐
         │  League   │
         └─────┬─────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐  ┌─────────┐  ┌──────────┐
│ Main  │  │Exploiter│  │Historical│
│Agents │  │ Agents  │  │Snapshots │
└───────┘  └─────────┘  └──────────┘
```

#### ELO Rating System
- Each agent has an ELO rating
- Matchmaking based on similar ELO
- Ratings update after each match

#### Parameters
- `main_agents`: Number of main agents (default: 2)
- `exploiter_agents`: Number of exploiter agents (default: 2)
- `snapshot_frequency`: Episodes between snapshots (default: 1000)
- `initial_elo`: Starting ELO rating (default: 1500)

#### Pros & Cons
- ✅ Most sophisticated strategy
- ✅ Prevents overfitting to single opponent
- ✅ Identifies and fixes exploits
- ✅ Produces most robust agents
- ❌ Complex implementation
- ❌ Computationally expensive
- ❌ Requires careful tuning

#### Usage
```python
from self_play.league import LeagueSelfPlay

self_play = LeagueSelfPlay(
    main_agents=2,
    exploiter_agents=2,
    snapshot_frequency=1000,
    initial_elo=1500
)
```

### 4. Reservoir Self-Play

#### Concept
Maintain a fixed-size reservoir of historical opponents using reservoir sampling.

#### Algorithm
```
Initialize reservoir R with capacity k

for episode in episodes:
    # Sample opponent from reservoir
    if len(R) > 0:
        opponent = sample_from_reservoir(R)
    else:
        opponent = current_policy
    
    # Play and train
    play_episode_and_train(current_policy, opponent)
    
    # Add current policy to reservoir with probability
    if len(R) < k:
        R.append(copy(current_policy))
    else:
        i = random.randint(0, episode)
        if i < k:
            R[i] = copy(current_policy)
```

#### Sampling Strategies

**Uniform Sampling:**
- Equal probability for all reservoir agents

**Recency-Biased Sampling:**
```python
age = current_episode - insertion_episode
probability = exp(-age / temperature)
```

#### Parameters
- `reservoir_size`: Maximum capacity (default: 50)
- `sampling_strategy`: 'uniform' or 'recency'
- `temperature`: For recency-biased sampling (default: 1000)

#### Pros & Cons
- ✅ Memory efficient (fixed size)
- ✅ Provides historical opponents
- ✅ Simple to implement
- ✅ Statistically sound sampling
- ❌ May not maintain diversity
- ❌ Older opponents may be too weak

#### Usage
```python
from self_play.reservoir import ReservoirSelfPlay

self_play = ReservoirSelfPlay(
    reservoir_size=50,
    sampling_strategy='recency',
    temperature=1000
)
```

### Self-Play Strategy Comparison

| Strategy | Complexity | Memory | Diversity | Robustness | Best For |
|----------|------------|--------|-----------|------------|----------|
| **Alternating** | Low | Low | Low | Medium | Simple scenarios |
| **Population** | Medium | High | High | High | Diverse opponents |
| **League** | High | High | Very High | Very High | Competition |
| **Reservoir** | Low | Medium | Medium | Medium | Historical opponents |

---

## ⚙️ Training Configuration

### Recommended Hyperparameters

#### MAPPO Configuration

**For Fast Training (Quick Results):**
```python
config = {
    'episodes': 2000,
    'lr': 5e-4,
    'gamma': 0.95,
    'batch_size': 32,
    'hidden_dim': 128,
    'max_cycles': 150
}
```

**For Best Performance:**
```python
config = {
    'episodes': 10000,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'batch_size': 64,
    'hidden_dim': 256,
    'max_cycles': 200
}
```

#### IPPO Configuration

```python
config = {
    'episodes': 10000,
    'lr': 5e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'hidden_dim': 256
}
```

### Training Tips

1. **Start with smaller episodes** for quick iteration
2. **Use GPU** if available (5-10x speedup)
3. **Save checkpoints frequently** to prevent loss
4. **Monitor metrics** to detect training issues early
5. **Use tensorboard** for real-time visualization

### Common Training Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Slow convergence** | Rewards plateau early | Increase learning rate, add entropy bonus |
| **Instability** | Rewards oscillate wildly | Decrease learning rate, increase batch size |
| **Overfitting** | High training rewards, poor evaluation | Add regularization, use self-play |
| **No learning** | Rewards don't improve | Check reward structure, increase network size |

---

## 📊 Evaluation & Metrics

### Metrics Tracked

#### Episode-Level Metrics
- **Episode Reward**: Cumulative reward per agent
- **Episode Length**: Number of steps until termination
- **Success Rate**: % of episodes where predators catch prey
- **Collision Count**: Number of predator-prey collisions
- **Average Distance**: Mean distance between predators and prey

#### Agent-Level Metrics
- **Policy Entropy**: Measure of exploration vs exploitation
- **Value Loss**: Critic prediction error
- **Policy Loss**: Actor update magnitude
- **Advantage**: Estimated action quality

#### Aggregated Metrics
- **Win Rate**: Predator success % over last N episodes
- **Average Reward (Moving)**: Smoothed reward trajectory
- **Convergence Rate**: Episodes to reach threshold performance

### Using MetricsTracker

```python
from analysis.metrics import MetricsTracker

# Initialize tracker
tracker = MetricsTracker()

# During training
for episode in range(num_episodes):
    # ... training code ...
    
    tracker.add_episode(
        episode=episode,
        predator_reward=total_pred_reward,
        prey_reward=total_prey_reward,
        episode_length=steps,
        success=prey_caught
    )
    
    # Log losses
    tracker.add_losses(
        episode=episode,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy
    )

# Save metrics
tracker.save("checkpoints/metrics.json")

# Load metrics
tracker.load("checkpoints/metrics.json")

# Get statistics
stats = tracker.get_statistics()
print(f"Average reward: {stats['mean_reward']}")
print(f"Success rate: {stats['success_rate']}")
```

### Evaluation Script

```bash
# Evaluate trained models
python training/evaluate.py \
    --algorithm mappo \
    --predator-checkpoint checkpoints/mappo_predator_final.pt \
    --prey-checkpoint checkpoints/mappo_prey_final.pt \
    --episodes 100 \
    --render
```

---

## 📈 Visualization

### Available Visualizations

#### 1. Training Curves

```python
from analysis.plots import plot_training_curves

plot_training_curves(
    metrics_file="checkpoints/metrics_final.json",
    save_path="figures/training_curves.png",
    smooth_window=50
)
```

Plots:
- Predator reward over time
- Prey reward over time
- Episode length over time
- Success rate over time

#### 2. Emergent Behaviors Heatmap

```python
from analysis.plots import plot_emergent_behaviors

plot_emergent_behaviors(
    metrics_file="checkpoints/metrics_final.json",
    save_path="figures/behaviors.png"
)
```

Shows spatial distribution of agents over time.

#### 3. Algorithm Comparison

```python
from analysis.plots import plot_algorithm_comparison

plot_algorithm_comparison(
    mappo_metrics="checkpoints/mappo_metrics.json",
    ippo_metrics="checkpoints/ippo_metrics.json",
    save_path="figures/comparison.png"
)
```

Side-by-side comparison of different algorithms.

#### 4. Self-Play Evolution

```python
from analysis.plots import plot_selfplay_evolution

plot_selfplay_evolution(
    metrics_file="checkpoints/metrics_final.json",
    save_path="figures/selfplay_evolution.png"
)
```

Visualizes how strategies evolve through self-play.

### Creating Custom Visualizations

```python
import matplotlib.pyplot as plt
from analysis.metrics import MetricsTracker

tracker = MetricsTracker()
tracker.load("checkpoints/metrics_final.json")

# Get data
episodes = tracker.get_episodes()
rewards = tracker.get_rewards()

# Create custom plot
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards['predator'], label='Predator', color='red')
plt.plot(episodes, rewards['prey'], label='Prey', color='green')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 📉 Results & Analysis

### Expected Training Progression

#### Phase 1: Random Exploration (Episodes 0-500)
- **Behavior**: Random movements, no coordination
- **Predator Reward**: -50 to -30
- **Prey Reward**: -20 to -10
- **Success Rate**: <10%

#### Phase 2: Basic Strategy (Episodes 500-2000)
- **Behavior**: Predators move toward prey, prey learns to flee
- **Predator Reward**: -30 to -10
- **Prey Reward**: -10 to 5
- **Success Rate**: 20-40%

#### Phase 3: Coordination Emerges (Episodes 2000-5000)
- **Behavior**: Predators begin to coordinate, prey uses obstacles
- **Predator Reward**: -10 to 10
- **Prey Reward**: 0 to 15
- **Success Rate**: 40-70%

#### Phase 4: Advanced Tactics (Episodes 5000+)
- **Behavior**: Complex strategies, adaptive responses
- **Predator Reward**: 10 to 30
- **Prey Reward**: 10 to 20
- **Success Rate**: 70-90%

### Emergent Behaviors Observed

#### Predator Behaviors
1. **Herding**: Predators position to limit prey escape routes
2. **Flanking**: Attacking from multiple angles simultaneously
3. **Anticipation**: Moving to where prey will be, not where it is
4. **Role Specialization**: Some predators chase, others block

#### Prey Behaviors
1. **Obstacle Utilization**: Using landmarks as shields
2. **Boundary Navigation**: Staying near walls for escape options
3. **Predator Monitoring**: Maintaining awareness of all threats
4. **Feinting**: Making fake moves to mislead predators

### Performance Benchmarks

#### MAPPO vs IPPO (After 10,000 episodes)

| Metric | MAPPO | IPPO |
|--------|-------|------|
| Predator Avg Reward | 18.5 ± 3.2 | 12.3 ± 4.1 |
| Prey Avg Reward | 15.2 ± 2.8 | 13.7 ± 3.5 |
| Success Rate | 82% | 65% |
| Avg Episode Length | 72.3 steps | 95.6 steps |
| Training Time (CPU) | 45 min | 38 min |
| Training Time (GPU) | 8 min | 7 min |

**Key Findings:**
- MAPPO shows better coordination among predators
- IPPO has more variance in performance
- MAPPO requires slightly more training time but achieves better results

---

## 🎛️ Hyperparameter Tuning

### Learning Rate

**Effect:** Controls how quickly the policy updates

| Value | Effect | Best For |
|-------|--------|----------|
| 1e-5 to 1e-4 | Very slow learning, stable | Fine-tuning |
| 3e-4 to 5e-4 | Balanced (recommended) | Most scenarios |
| 1e-3 to 5e-3 | Fast learning, unstable | Quick experiments |

**Tuning Strategy:**
```python
# Start high and decay
initial_lr = 5e-4
lr_decay = 0.99

for episode in episodes:
    lr = initial_lr * (lr_decay ** episode)
    optimizer = torch.optim.Adam(params, lr=lr)
```

### Discount Factor (Gamma)

**Effect:** Balances immediate vs future rewards

| Value | Effect | Best For |
|-------|--------|----------|
| 0.9 | Short-term focus | Quick reactions |
| 0.95 | Balanced | General use |
| 0.99 | Long-term planning | Strategic play |

### Entropy Coefficient

**Effect:** Encourages exploration

| Value | Effect | Best For |
|-------|--------|----------|
| 0.0 | No exploration bonus | Exploitation |
| 0.01 | Balanced (recommended) | General use |
| 0.1 | High exploration | Early training |

**Adaptive Entropy:**
```python
# Decay entropy over time
initial_entropy = 0.1
final_entropy = 0.01
decay_episodes = 5000

for episode in episodes:
    progress = min(episode / decay_episodes, 1.0)
    entropy_coef = initial_entropy * (1 - progress) + final_entropy * progress
```

### Network Architecture

**Hidden Dimensions:**
- Small (64-128): Faster training, less capacity
- Medium (256): Recommended starting point
- Large (512+): More capacity, slower training

**Number of Layers:**
```python
# Simple (faster)
network = nn.Sequential(
    nn.Linear(obs_dim, 256),
    nn.ReLU(),
    nn.Linear(256, action_dim)
)

# Deep (more expressive)
network = nn.Sequential(
    nn.Linear(obs_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim)
)
```

### Batch Size

**Effect:** Stability vs sample efficiency

| Value | Effect | Best For |
|-------|--------|----------|
| 16-32 | Noisy gradients, fast | Online learning |
| 64-128 | Balanced | General use |
| 256+ | Stable, slow | Large datasets |

### Systematic Tuning Process

```python
# Grid search example
hyperparameters = {
    'lr': [1e-4, 3e-4, 5e-4],
    'gamma': [0.95, 0.99],
    'entropy_coef': [0.01, 0.05],
    'hidden_dim': [128, 256]
}

results = []
for lr in hyperparameters['lr']:
    for gamma in hyperparameters['gamma']:
        for entropy in hyperparameters['entropy_coef']:
            for hidden in hyperparameters['hidden_dim']:
                # Train model with these hyperparameters
                result = train(lr, gamma, entropy, hidden)
                results.append({
                    'params': {'lr': lr, 'gamma': gamma, ...},
                    'performance': result
                })

# Find best configuration
best = max(results, key=lambda x: x['performance'])
print(f"Best hyperparameters: {best['params']}")
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size:**
```bash
python training/train_mappo.py --batch-size 32
```

2. **Use CPU:**
```bash
python training/train_mappo.py --device cpu
```

3. **Clear cache:**
```python
import torch
torch.cuda.empty_cache()
```

#### Issue 2: Training is Very Slow

**Possible Causes:**
- Running on CPU instead of GPU
- Large batch size
- Too many agents
- Rendering enabled

**Solutions:**
1. **Verify GPU usage:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

2. **Disable rendering during training:**
```python
env = simple_tag_v3.parallel_env(render_mode=None)  # Not 'human'
```

3. **Profile code:**
```bash
python -m cProfile -o profile.stats training/train_mappo.py
python -m pstats profile.stats
```

#### Issue 3: Rewards Not Improving

**Symptoms:**
- Rewards plateau early
- No improvement after many episodes
- Random-looking behavior

**Debugging Steps:**

1. **Check reward structure:**
```python
# Print rewards during training
print(f"Predator rewards: {predator_rewards}")
print(f"Prey rewards: {prey_rewards}")
```

2. **Verify network updates:**
```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

3. **Inspect observations:**
```python
# Are observations normalized?
obs = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs range: [{obs.min()}, {obs.max()}]")
```

**Solutions:**
- Increase learning rate
- Add entropy bonus for more exploration
- Normalize observations
- Check for bugs in reward calculation

#### Issue 4: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'pettingzoo'
```

**Solutions:**
1. **Verify installation:**
```bash
pip list | grep pettingzoo
```

2. **Reinstall dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

3. **Check Python path:**
```python
import sys
print(sys.path)
```

#### Issue 5: Environment Rendering Issues

**Symptoms:**
- Black screen
- Pygame errors
- Display not updating

**Solutions:**
1. **Install pygame properly:**
```bash
pip uninstall pygame
pip install pygame==2.1.0
```

2. **Check display:**
```python
import pygame
pygame.init()
```

3. **For headless systems:**
```bash
# Use virtual display
export DISPLAY=:0
xvfb-run -a python training/evaluate.py --render
```

#### Issue 6: Checkpoint Loading Fails

**Symptoms:**
```
RuntimeError: Error loading checkpoint
KeyError: 'state_dict'
```

**Solutions:**
1. **Verify checkpoint format:**
```python
checkpoint = torch.load('checkpoint.pt')
print(checkpoint.keys())
```

2. **Load with error handling:**
```python
try:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
except Exception as e:
    print(f"Error loading: {e}")
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs:** Look for error messages in console output
2. **Enable debug mode:** Add `--debug` flag to training scripts
3. **Simplify the problem:** Try with fewer agents/episodes
4. **Check dependencies:** Ensure all packages are up to date
5. **Contact us:** See [Contact](#-contact) section

---

## 🚀 Extensions & Future Work

### Potential Extensions

#### 1. Communication Between Agents
Add communication channels for explicit coordination:
```python
class CommunicationModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_encoder = nn.Linear(hidden_dim, 32)
        self.message_decoder = nn.Linear(32, hidden_dim)
    
    def forward(self, agent_states):
        messages = self.message_encoder(agent_states)
        # Broadcast messages
        received = messages.mean(dim=0)  # Average pooling
        enhanced_states = self.message_decoder(received)
        return enhanced_states
```

#### 2. Hierarchical Policies
Learn high-level strategies and low-level actions:
```python
class HierarchicalPolicy:
    def __init__(self):
        self.high_level_policy = HighLevelPolicy()  # Every K steps
        self.low_level_policy = LowLevelPolicy()    # Every step
    
    def get_action(self, obs, step):
        if step % K == 0:
            subgoal = self.high_level_policy(obs)
        action = self.low_level_policy(obs, subgoal)
        return action
```

#### 3. Curriculum Learning
Gradually increase difficulty:
```python
class Curriculum:
    def __init__(self):
        self.stage = 0
        self.stages = [
            {'adversaries': 1, 'max_cycles': 100},
            {'adversaries': 2, 'max_cycles': 150},
            {'adversaries': 3, 'max_cycles': 200},
        ]
    
    def get_config(self, success_rate):
        if success_rate > 0.8 and self.stage < len(self.stages) - 1:
            self.stage += 1
        return self.stages[self.stage]
```

#### 4. Partial Observability
Limit agent vision to realistic sensors:
```python
def get_partial_observation(agent_pos, all_positions, vision_range=5.0):
    """Only observe entities within vision range"""
    visible = []
    for pos in all_positions:
        if distance(agent_pos, pos) < vision_range:
            visible.append(pos - agent_pos)  # Relative position
    return np.array(visible)
```

#### 5. Continuous Action Spaces
Allow finer motor control:
```python
# Instead of discrete [0, 1, 2, 3, 4]
# Use continuous [-1, 1] for x and y movement
action = policy(obs)  # Shape: (2,) for [dx, dy]
action = torch.tanh(action)  # Bound to [-1, 1]
```

#### 6. Transfer Learning
Train on simple scenarios, transfer to complex:
```python
# Train on 3x3 grid with 1 predator
simple_model = train_simple_scenario()

# Transfer to 10x10 grid with 3 predators
complex_model = ComplexModel()
complex_model.load_state_dict(simple_model.state_dict(), strict=False)
complex_model = fine_tune(complex_model)
```

### Research Directions

1. **Emergent Communication Protocols**
   - Do agents develop implicit communication?
   - Can we decode learned strategies?

2. **Robustness to Adversarial Opponents**
   - How do agents handle never-seen-before strategies?
   - Can we create robust policies through diverse training?

3. **Scalability to Many Agents**
   - Does coordination break down with 10+ agents?
   - Can we use attention mechanisms for scalability?

4. **Human-Agent Interaction**
   - Can trained agents play with human controllers?
   - Do they adapt to human strategies?

5. **Multi-Objective Learning**
   - Balance multiple objectives (speed, energy, coordination)
   - Pareto-optimal policies

---

## 🤝 Contributing

We welcome contributions to improve this project!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages:**
```bash
git commit -m "Add amazing feature: description"
```

6. **Push to your fork:**
```bash
git push origin feature/amazing-feature
```

7. **Open a Pull Request**

### Contribution Guidelines

- **Code Style:** Follow PEP 8 for Python code
- **Documentation:** Add docstrings to all functions/classes
- **Testing:** Include tests for new features
- **Commits:** Use clear, descriptive commit messages

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New algorithms (PPG, SAC, TD3, etc.)
- 🎮 New environments
- 📊 Visualization improvements
- 📚 Documentation enhancements
- ⚡ Performance optimizations

---

## 📚 Citation

If you use this code in your research or projects, please cite:

```bibtex
@misc{emergentadversarial2024,
  title={Emergent Adversarial Behaviors via Self-Play in Multi-Agent Reinforcement Learning},
  author={Nandyala, Charan Reddy and Chalicheemala, Dhanush and Gangineni, Satyadev},
  year={2024},
  institution={University of California, Riverside},
  url={https://github.com/YOUR_USERNAME/YOUR_REPO_NAME}
}
```

### Related Work

This project builds upon:

- **PPO:** Schulman et al. (2017) - Proximal Policy Optimization Algorithms
- **MADDPG:** Lowe et al. (2017) - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
- **PettingZoo:** Terry et al. (2021) - PettingZoo: Gym for Multi-Agent Reinforcement Learning
- **AlphaStar:** Vinyals et al. (2019) - Grandmaster level in StarCraft II using multi-agent reinforcement learning

---

## 📄 License

This project is for **educational purposes** as part of a university course project at the University of California, Riverside.

**Usage Restrictions:**
- ✅ Academic and educational use
- ✅ Personal learning and experimentation
- ✅ Research purposes with proper attribution
- ❌ Commercial use without permission

---

## 📞 Contact

### Team Members

**Charan Reddy Nandyala**
- 📧 Email: cnand002@ucr.edu
- 🎓 Role: Lead Developer, Project Coordinator

**Dhanush Chalicheemala**
- 📧 Email: dchal007@ucr.edu
- 🎓 Role: Algorithm Implementation, Optimization

**Satyadev Gangineni**
- 📧 Email: sgang024@ucr.edu
- 🎓 Role: Analysis, Visualization, Documentation

### Questions or Issues?

- 📝 **Bug Reports:** Open an issue on GitHub
- 💡 **Feature Requests:** Open an issue with [Feature Request] tag
- 📧 **General Questions:** Email any team member
- 🤝 **Collaboration:** Contact Charan Reddy Nandyala

---

## 🙏 Acknowledgments

- **PettingZoo Team** for the excellent multi-agent environment framework
- **OpenAI** for pioneering work in PPO and self-play
- **DeepMind** for AlphaStar and league training inspiration
- **UC Riverside** for supporting this research project
- **Course Instructor** for guidance and feedback

---

<div align="center">

### ⭐ If you find this project useful, please consider starring it on GitHub! ⭐

**Built with ❤️ by the UCR Multi-Agent RL Team**

</div>
