"""
Independent Proximal Policy Optimization (IPPO).
Each agent learns independently with its own actor-critic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List


class ActorCritic(nn.Module):
    """Combined actor-critic network for IPPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        features = self.shared(obs)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        action_probs, _ = self.forward(obs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action


class IPPO:
    """
    Independent PPO where each agent has its own actor-critic network.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        lr: float = 5e-4,  # Optimized: increased from 3e-4
        gamma: float = 0.98,  # Optimized: reduced from 0.99
        clip_param: float = 0.3,  # Optimized: increased from 0.2
        value_coef: float = 1.0,  # Optimized: increased from 0.5
        entropy_coef: float = 0.05,  # Optimized: increased from 0.01
        max_grad_norm: float = 1.0,  # Optimized: increased from 0.5
        hidden_dim: int = 256,  # Optimized: increased from 128
        device: str = "cpu"
    ):
        """
        Initialize IPPO algorithm.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            n_agents: Number of agents
            lr: Learning rate (default: 5e-4, optimized for Simple Tag)
            gamma: Discount factor (default: 0.98)
            clip_param: PPO clipping parameter (default: 0.3)
            value_coef: Value loss coefficient (default: 1.0)
            entropy_coef: Entropy bonus coefficient (default: 0.05)
            max_grad_norm: Maximum gradient norm (default: 1.0)
            hidden_dim: Hidden layer dimension (default: 256)
            device: Device for computation
        """
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Create independent actor-critic for each agent with optimized hidden dimension
        self.agent_networks = [
            ActorCritic(obs_dim, action_dim, hidden_dim).to(device) for _ in range(n_agents)
        ]
        
        # Optimizers
        self.optimizers = [
            optim.Adam(network.parameters(), lr=lr) 
            for network in self.agent_networks
        ]
        
        self.buffers = [[] for _ in range(n_agents)]
    
    def select_actions(self, observations: List[np.ndarray], deterministic: bool = False):
        """Select actions for all agents independently."""
        actions = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.agent_networks[i].get_action(obs_tensor, deterministic)
            actions.append(action.item())
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transition in each agent's buffer."""
        for i in range(self.n_agents):
            self.buffers[i].append((
                obs[i], actions[i], rewards[i], next_obs[i], dones[i]
            ))
    
    def update(self, n_epochs: int = 10):
        """Update each agent's policy independently.
        
        Args:
            n_epochs: Number of update epochs (default: 10, optimized from 4)
        """
        total_losses = {"actor_loss": 0, "critic_loss": 0, "entropy": 0}
        
        for agent_id in range(self.n_agents):
            if len(self.buffers[agent_id]) == 0:
                continue
            
            # Convert buffer to tensors
            obs_list, action_list, reward_list, next_obs_list, done_list = zip(*self.buffers[agent_id])
            
            obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
            action_batch = torch.LongTensor(np.array(action_list)).to(self.device)
            reward_batch = torch.FloatTensor(np.array(reward_list)).to(self.device)
            done_batch = torch.FloatTensor(np.array(done_list)).to(self.device)
            
            # Compute returns
            returns = self._compute_returns(reward_batch, done_batch)
            
            # Get old values and log probs
            with torch.no_grad():
                old_probs, old_values = self.agent_networks[agent_id](obs_batch)
                old_log_probs = torch.log(old_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-8)
                advantages = returns - old_values.squeeze(-1)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for epoch in range(n_epochs):
                probs, values = self.agent_networks[agent_id](obs_batch)
                log_probs = torch.log(probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-8)
                
                # Actor loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(values.squeeze(-1), returns)
                
                # Entropy bonus
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizers[agent_id].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent_networks[agent_id].parameters(), self.max_grad_norm)
                self.optimizers[agent_id].step()
                
                total_losses["actor_loss"] += actor_loss.item()
                total_losses["critic_loss"] += critic_loss.item()
                total_losses["entropy"] += entropy.item()
            
            # Clear buffer
            self.buffers[agent_id].clear()
        
        # Average losses
        n_updates = self.n_agents * n_epochs
        return {k: v / n_updates for k, v in total_losses.items()}
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        
        return returns
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'agents': [network.state_dict() for network in self.agent_networks]
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        for i, agent_state in enumerate(checkpoint['agents']):
            self.agent_networks[i].load_state_dict(agent_state)

