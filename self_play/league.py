"""
League Self-Play Strategy.
Inspired by AlphaStar's league training with main agents, exploiters, and historical players.
"""

import copy
import random
from typing import List, Dict, Any
from collections import deque


class LeaguePlayer:
    """Represents a player in the league."""
    
    def __init__(self, agent, player_type: str, generation: int = 0):
        """
        Initialize a league player.
        
        Args:
            agent: The RL agent
            player_type: Type of player ("main", "exploiter", "historical")
            generation: Generation/version number
        """
        self.agent = agent
        self.player_type = player_type
        self.generation = generation
        self.wins = 0
        self.games = 0
        self.elo_rating = 1000  # Initial ELO rating
    
    def get_win_rate(self) -> float:
        """Calculate win rate."""
        if self.games == 0:
            return 0.0
        return self.wins / self.games
    
    def update_elo(self, opponent_elo: float, result: float):
        """
        Update ELO rating.
        
        Args:
            opponent_elo: Opponent's ELO rating
            result: 1.0 for win, 0.5 for draw, 0.0 for loss
        """
        K = 32  # K-factor
        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        self.elo_rating += K * (result - expected)


class LeagueSelfPlay:
    """
    League-based self-play with multiple player types and matchmaking.
    """
    
    def __init__(
        self,
        predator_agent,
        prey_agent,
        n_main_agents: int = 2,
        n_exploiters: int = 1,
        historical_window: int = 10,
        checkpoint_interval: int = 1000,
        matchmaking_strategy: str = "elo"  # "elo", "random", "prioritized"
    ):
        """
        Initialize league self-play.
        
        Args:
            predator_agent: Initial predator agent
            prey_agent: Initial prey agent
            n_main_agents: Number of main learner agents
            n_exploiters: Number of exploiter agents
            historical_window: How many historical snapshots to keep
            checkpoint_interval: Steps between creating historical snapshots
            matchmaking_strategy: How to match opponents
        """
        self.checkpoint_interval = checkpoint_interval
        self.matchmaking_strategy = matchmaking_strategy
        self.historical_window = historical_window
        
        # Initialize predator league
        self.predator_league = {
            'main': [LeaguePlayer(copy.deepcopy(predator_agent), "main", gen=i) 
                    for i in range(n_main_agents)],
            'exploiter': [LeaguePlayer(copy.deepcopy(predator_agent), "exploiter", gen=i)
                         for i in range(n_exploiters)],
            'historical': deque(maxlen=historical_window)
        }
        
        # Initialize prey league
        self.prey_league = {
            'main': [LeaguePlayer(copy.deepcopy(prey_agent), "main", gen=i)
                    for i in range(n_main_agents)],
            'exploiter': [LeaguePlayer(copy.deepcopy(prey_agent), "exploiter", gen=i)
                         for i in range(n_exploiters)],
            'historical': deque(maxlen=historical_window)
        }
        
        # Current active players
        self.current_predator = self.predator_league['main'][0]
        self.current_prey = self.prey_league['main'][0]
        
        self.steps = 0
        self.generation = 0
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to create historical snapshots."""
        return self.steps % self.checkpoint_interval == 0 and self.steps > 0
    
    def create_checkpoint(self):
        """Add current main agents to historical pool."""
        self.generation += 1
        
        # Checkpoint best performing main agent for each side
        best_predator = max(self.predator_league['main'], 
                           key=lambda x: x.elo_rating)
        best_prey = max(self.prey_league['main'],
                       key=lambda x: x.elo_rating)
        
        self.predator_league['historical'].append(
            LeaguePlayer(copy.deepcopy(best_predator.agent), "historical", self.generation)
        )
        self.prey_league['historical'].append(
            LeaguePlayer(copy.deepcopy(best_prey.agent), "historical", self.generation)
        )
        
        print(f"Step {self.steps}: Created checkpoint (Generation {self.generation})")
        print(f"  Best Predator ELO: {best_predator.elo_rating:.1f}")
        print(f"  Best Prey ELO: {best_prey.elo_rating:.1f}")
    
    def select_opponent(self, league: Dict) -> LeaguePlayer:
        """Select opponent from league based on matchmaking strategy."""
        # Collect all potential opponents
        all_players = (league['main'] + league['exploiter'] + 
                      list(league['historical']))
        
        if len(all_players) == 0:
            return league['main'][0]
        
        if self.matchmaking_strategy == "random":
            return random.choice(all_players)
        
        elif self.matchmaking_strategy == "elo":
            # Sample based on ELO (prefer similarly rated opponents)
            current_elo = self.current_predator.elo_rating
            weights = [1.0 / (1.0 + abs(p.elo_rating - current_elo) / 100.0) 
                      for p in all_players]
            return random.choices(all_players, weights=weights)[0]
        
        elif self.matchmaking_strategy == "prioritized":
            # Prioritize main agents and exploiters
            weights = []
            for p in all_players:
                if p.player_type == "main":
                    weights.append(3.0)
                elif p.player_type == "exploiter":
                    weights.append(2.0)
                else:  # historical
                    weights.append(1.0)
            return random.choices(all_players, weights=weights)[0]
        
        return random.choice(all_players)
    
    def step(self, observations, deterministic=False):
        """Select actions for current active players."""
        self.steps += 1
        
        if self.should_checkpoint():
            self.create_checkpoint()
        
        # Get actions from current players
        predator_actions = self.current_predator.agent.select_actions(
            observations['predators'],
            deterministic=deterministic
        )
        prey_actions = self.current_prey.agent.select_actions(
            observations['prey'],
            deterministic=deterministic
        )
        
        return {
            'predators': predator_actions,
            'prey': prey_actions
        }
    
    def update(self):
        """Update main agents and exploiters."""
        metrics = {}
        
        # Update main agents
        for i, player in enumerate(self.predator_league['main']):
            metrics[f'predator_main_{i}'] = player.agent.update()
        
        for i, player in enumerate(self.prey_league['main']):
            metrics[f'prey_main_{i}'] = player.agent.update()
        
        # Update exploiters
        for i, player in enumerate(self.predator_league['exploiter']):
            metrics[f'predator_exploiter_{i}'] = player.agent.update()
        
        for i, player in enumerate(self.prey_league['exploiter']):
            metrics[f'prey_exploiter_{i}'] = player.agent.update()
        
        return metrics
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transitions for all active learners."""
        # Store for main agents
        for player in self.predator_league['main']:
            player.agent.store_transition(
                obs['predators'], actions['predators'],
                rewards['predators'], next_obs['predators'],
                dones['predators']
            )
        
        for player in self.prey_league['main']:
            player.agent.store_transition(
                obs['prey'], actions['prey'],
                rewards['prey'], next_obs['prey'],
                dones['prey']
            )
        
        # Update game statistics
        self.current_predator.games += 1
        self.current_prey.games += 1
        
        # Determine winner based on rewards
        avg_predator_reward = sum(rewards['predators']) / len(rewards['predators'])
        avg_prey_reward = sum(rewards['prey']) / len(rewards['prey'])
        
        if avg_predator_reward > avg_prey_reward:
            self.current_predator.wins += 1
            self.current_predator.update_elo(self.current_prey.elo_rating, 1.0)
            self.current_prey.update_elo(self.current_predator.elo_rating, 0.0)
        elif avg_prey_reward > avg_predator_reward:
            self.current_prey.wins += 1
            self.current_prey.update_elo(self.current_predator.elo_rating, 1.0)
            self.current_predator.update_elo(self.current_prey.elo_rating, 0.0)
        else:
            # Draw
            self.current_predator.update_elo(self.current_prey.elo_rating, 0.5)
            self.current_prey.update_elo(self.current_predator.elo_rating, 0.5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get league statistics."""
        return {
            'total_steps': self.steps,
            'generation': self.generation,
            'predator_main_avg_elo': sum(p.elo_rating for p in self.predator_league['main']) / 
                                     len(self.predator_league['main']),
            'prey_main_avg_elo': sum(p.elo_rating for p in self.prey_league['main']) / 
                                 len(self.prey_league['main']),
            'current_predator_elo': self.current_predator.elo_rating,
            'current_prey_elo': self.current_prey.elo_rating
        }

