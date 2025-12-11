# Multi-Agent Predator-Prey Self-Play in Reinforcement Learning

A comparative study of MAPPO (Multi-Agent Proximal Policy Optimization) and IPPO (Independent Proximal Policy Optimization) algorithms in adversarial multi-agent reinforcement learning scenarios using competitive self-play.

## 📋 Project Overview

This project investigates emergent adversarial behaviors in multi-agent reinforcement learning through competitive self-play mechanisms in predator-prey environments. We compare two prominent MARL algorithms (MAPPO and IPPO) across three distinct self-play strategies: alternating, population-based, and league-based training.

### Key Findings

- **IPPO Predators**: Consistently achieve 96-98% win rates across all strategies
- **MAPPO**: Exhibits more balanced dynamics with better prey survival, achieving positive prey rewards (+3.1) in league-based self-play
- **Trade-offs**: IPPO excels in fast convergence and scalability, while MAPPO provides better coordination and balanced gameplay

## 🏗️ Project Structure

```
.
├── algorithms/          # MARL algorithm implementations
│   ├── mappo.py        # Multi-Agent PPO with centralized critic
│   └── ippo.py         # Independent PPO (fully decentralized)
├── envs/               # Environment implementations
│   ├── custom_predator_prey.py      # Custom gridworld environment
│   └── pettingzoo_prey_predator.py # Environment wrapper
├── self_play/          # Self-play training strategies
│   ├── alternating.py  # Alternating self-play
│   ├── population.py   # Population-based self-play
│   └── league.py       # League-based self-play (AlphaStar-inspired)
├── training/           # Training scripts
│   ├── compare_mappo_vs_ippo.py           # Main comparison script
│   ├── compare_mappo_vs_ippo_league.py    # League-based comparison
│   ├── compare_mappo_vs_ippo_population.py # Population-based comparison
│   ├── train_mappo_optimized.py          # Standalone MAPPO training
│   └── train_ippo_optimized.py           # Standalone IPPO training
├── analysis/           # Metrics and visualization
│   ├── metrics.py      # Performance tracking
│   └── plots.py        # Plotting utilities
├── config/             # Configuration files
│   └── hyperparameters.py
├── results/            # Experimental results and visualizations
└── ieee_report_template.tex  # IEEE format research paper template
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- (Optional) CUDA for GPU acceleration

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Charanreddy1611/multiagent-predator-prey-selfplay-in-RL.git
cd multiagent-predator-prey-selfplay-in-RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Running Comparisons

#### Main Comparison (Alternating Self-Play)
```bash
python training/compare_mappo_vs_ippo.py --adversaries 2 --good 2 --episodes 2000
```

#### League-Based Self-Play Comparison
```bash
python training/compare_mappo_vs_ippo_league.py --adversaries 2 --good 2 --episodes 2000
```

#### Population-Based Self-Play Comparison
```bash
python training/compare_mappo_vs_ippo_population.py --adversaries 2 --good 2 --episodes 2000
```

### Training Individual Algorithms

#### Train MAPPO
```bash
python training/train_mappo_optimized.py --adversaries 2 --good 2 --episodes 5000
```

#### Train IPPO
```bash
python training/train_ippo_optimized.py --adversaries 2 --good 2 --episodes 5000
```

### Command Line Arguments

- `--adversaries`: Number of predators (default: 2)
- `--good`: Number of prey (default: 2)
- `--episodes`: Number of training episodes (default: 2000)
- `--max-cycles`: Maximum steps per episode (default: 200)
- `--device`: Device for computation (default: "cuda" if available, else "cpu")
- `--save-dir`: Directory to save results (default: "comparison_results")

## 🎮 Environment

### Custom Predator-Prey Gridworld

- **Grid Size**: 20×20 continuous space
- **Max Steps**: 200 per episode
- **Agent Configurations**: 2v2 (2 predators vs 2 prey) and 3v2 (3 predators vs 2 prey)
- **Capture Radius**: 1.5 units
- **Obstacles**: 3 fixed obstacles per episode

### Reward Structure

**Predators:**
- +10.0 for successfully tagging prey
- +0.2 for proximity to prey (within 3 units)

**Prey:**
- +0.1 survival bonus per step
- -10.0 penalty when tagged
- +0.3 distance reward when all predators are >3 units away

## 🧠 Algorithms

### MAPPO (Multi-Agent Proximal Policy Optimization)
- **Architecture**: Centralized critic with decentralized actors
- **Training**: Centralized Training, Decentralized Execution (CTDE)
- **Key Feature**: Shared value function using global state
- **Best For**: Scenarios requiring coordination and balanced gameplay

### IPPO (Independent Proximal Policy Optimization)
- **Architecture**: Fully independent actor-critic networks per agent
- **Training**: Completely decentralized, no information sharing
- **Key Feature**: Robust to non-stationarity, better scalability
- **Best For**: Fast convergence, scenarios with numerical advantage

## 🎯 Self-Play Strategies

### 1. Alternating Self-Play
- Agents alternate between training and being frozen as opponents
- Switch interval: 500 steps
- **Pros**: Simple, stable training dynamics
- **Cons**: Sequential learning may be slower

### 2. Population-Based Self-Play
- Maintains diverse populations of historical agent snapshots
- Population size: 5 agents
- **Pros**: Robust, diverse opponents, prevents overfitting
- **Cons**: Memory intensive

### 3. League-Based Self-Play
- AlphaStar-inspired competitive training
- Main agents, exploiters, and historical players
- ELO-based matchmaking
- **Pros**: Most sophisticated, prevents overfitting
- **Cons**: Complex implementation, computationally expensive

## 📊 Results

### Key Metrics

- **IPPO Predator Win Rate**: 96-98% across all strategies
- **MAPPO Predator Win Rate**: 45-77% (more balanced)
- **Best Prey Performance**: MAPPO achieves +3.1 reward in league-based self-play

Detailed results tables and training curves are available in the `results/` directory and in the research paper template (`ieee_report_template.tex`).

## 📝 Research Paper

The project includes a complete IEEE format research paper template (`ieee_report_template.tex`) documenting:
- Algorithm architectures and training procedures
- Experimental methodology
- Comprehensive results and analysis
- Discussion of trade-offs between MAPPO and IPPO




## 📄 License

This project is for educational and research purposes.


## 📚 References

- Yu, C., et al. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS.
- Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature.

## 🔗 Related Work

This project extends research on:
- Multi-agent reinforcement learning
- Self-play training strategies
- Competitive MARL scenarios
- Centralized vs. independent learning approaches

---

For questions or issues, please open an issue on GitHub.

