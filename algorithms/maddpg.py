"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
Centralized training with decentralized execution for continuous/discrete actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from collections import deque
import random


class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # For discrete actions
        )
    
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Centralized critic network for MADDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        total_input = state_dim * n_agents + action_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        """
        Args:
            states: [batch, n_agents, obs_dim]
            actions: [batch, n_agents, action_dim]
        """
        # Flatten states and actions
        states_flat = states.reshape(states.shape[0], -1)
        actions_flat = actions.reshape(actions.shape[0], -1)
        x = torch.cat([states_flat, actions_flat], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for MADDPG."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)


class MADDPG:
    """
    Multi-Agent DDPG with centralized critic and decentralized actors.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize MADDPG algorithm.
        
        Args:
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            n_agents: Number of agents
            lr_actor: Learning rate for actors
            lr_critic: Learning rate for critics
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            device: Device for computation
        """
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim
        
        # Create actors and critics for each agent
        self.actors = [Actor(obs_dim, action_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim, action_dim).to(device) for _ in range(n_agents)]
        
        self.critics = [Critic(obs_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        self.target_critics = [Critic(obs_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        
        # Copy parameters to target networks
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics
        ]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_actions(self, observations: List[np.ndarray], epsilon: float = 0.0, deterministic: bool = False):
        """Select actions for all agents with epsilon-greedy exploration."""
        actions = []
        for i, obs in enumerate(observations):
            if not deterministic and random.random() < epsilon:
                # Random action
                action = random.randint(0, self.action_dim - 1)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = self.actors[i](obs_tensor)
                    if deterministic:
                        action = torch.argmax(action_probs, dim=-1).item()
                    else:
                        action = torch.distributions.Categorical(action_probs).sample().item()
            actions.append(action)
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transition in replay buffer."""
        self.replay_buffer.push(obs, actions, rewards, next_obs, dones)
    
    def update(self):
        """Update all agents using MADDPG."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)  # [batch, n_agents, obs_dim]
        actions = torch.LongTensor(actions).to(self.device)  # [batch, n_agents]
        rewards = torch.FloatTensor(rewards).to(self.device)  # [batch, n_agents]
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Convert actions to one-hot for critic input
        actions_onehot = torch.nn.functional.one_hot(actions, self.action_dim).float()
        
        total_critic_loss = 0
        total_actor_loss = 0
        
        for agent_id in range(self.n_agents):
            # Update critic
            with torch.no_grad():
                # Get target actions
                target_actions = []
                for i in range(self.n_agents):
                    target_action_probs = self.target_actors[i](next_states[:, i, :])
                    target_actions.append(target_action_probs)
                target_actions = torch.stack(target_actions, dim=1)  # [batch, n_agents, action_dim]
                
                # Compute target Q-value
                target_q = self.target_critics[agent_id](next_states, target_actions).squeeze(-1)
                target_q = rewards[:, agent_id] + self.gamma * target_q * (1 - dones[:, agent_id])
            
            # Current Q-value
            current_q = self.critics[agent_id](states, actions_onehot).squeeze(-1)
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
            self.critic_optimizers[agent_id].step()
            
            total_critic_loss += critic_loss.item()
            
            # Update actor
            # Get current actions
            current_actions = []
            for i in range(self.n_agents):
                if i == agent_id:
                    current_actions.append(self.actors[i](states[:, i, :]))
                else:
                    with torch.no_grad():
                        current_actions.append(self.actors[i](states[:, i, :]))
            current_actions = torch.stack(current_actions, dim=1)
            
            # Actor loss (policy gradient)
            actor_loss = -self.critics[agent_id](states, current_actions).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)
            self.actor_optimizers[agent_id].step()
            
            total_actor_loss += actor_loss.item()
        
        # Soft update target networks
        self._soft_update()
        
        return {
            "critic_loss": total_critic_loss / self.n_agents,
            "actor_loss": total_actor_loss / self.n_agents
        }
    
    def _soft_update(self):
        """Soft update target networks."""
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                          self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critics[i].parameters(), 
                                          self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics]
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['critics'][i])

