"""
Multi-Agent Proximal Policy Optimization (MAPPO).
Centralized training with decentralized execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple


class Actor(nn.Module):
    """Policy network for MAPPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        return self.net(obs)
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        probs = self.forward(obs)
        # Clamp probabilities to prevent NaN
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        # Renormalize after clamping
        probs = probs / probs.sum(dim=-1, keepdim=True)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action
    
    def get_log_prob(self, obs, action):
        """Get log probability of action."""
        probs = self.forward(obs)
        # Clamp probabilities to prevent NaN
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        # Renormalize after clamping
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)


class Critic(nn.Module):
    """Centralized value function for MAPPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class MAPPO:
    """
    Multi-Agent PPO with centralized critic and decentralized actors.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        lr: float = 5e-4,  # Optimized: increased from 1e-4
        gamma: float = 0.98,  # Optimized: reduced from 0.99
        clip_param: float = 0.3,  # Optimized: increased from 0.2
        value_coef: float = 1.0,  # Optimized: increased from 0.5
        entropy_coef: float = 0.05,  # Optimized: increased from 0.01
        max_grad_norm: float = 1.0,  # Optimized: increased from 0.5
        hidden_dim: int = 256,  # Optimized: increased from 128
        device: str = "cpu"
    ):
        """
        Initialize MAPPO algorithm.
        
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
        
        # Create actor for each agent with optimized hidden dimension
        self.actors = [Actor(obs_dim, action_dim, hidden_dim).to(device) for _ in range(n_agents)]
        
        # Centralized critic (takes global state)
        state_dim = obs_dim * n_agents
        self.critic = Critic(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = []
    
    def select_actions(self, observations: List[np.ndarray], deterministic: bool = False):
        """Select actions for all agents."""
        actions = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actors[i].get_action(obs_tensor, deterministic)
            actions.append(action.item())
        return actions
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store transition in buffer."""
        self.buffer.append((obs, actions, rewards, next_obs, dones))
    
    def update(self, n_epochs: int = 10):
        """Update policy using collected experiences.
        
        Args:
            n_epochs: Number of update epochs (default: 10, optimized from 4)
        """
        if len(self.buffer) == 0:
            return {}
        
        # Convert buffer to tensors
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for transition in self.buffer:
            obs, actions, rewards, next_obs, dones = transition
            obs_batch.append(obs)
            action_batch.append(actions)
            reward_batch.append(rewards)
            next_obs_batch.append(next_obs)
            done_batch.append(dones)
        
        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # Average rewards across agents for team-based learning
        team_rewards = reward_batch.mean(dim=1)  # [batch_size]
        team_dones = done_batch.max(dim=1)[0]  # [batch_size]
        
        # Compute returns for team
        returns = self._compute_returns(team_rewards, team_dones)
        
        # Global state for critic
        global_obs = obs_batch.reshape(obs_batch.shape[0], -1)
        values = self.critic(global_obs).squeeze(-1)
        advantages = returns - values.detach()
        
        # Normalize advantages (handle edge cases)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(n_epochs):
            for agent_id in range(self.n_agents):
                # Actor update
                obs_agent = obs_batch[:, agent_id, :]
                actions_agent = action_batch[:, agent_id]
                
                old_log_probs = self.actors[agent_id].get_log_prob(obs_agent, actions_agent).detach()
                new_log_probs = self.actors[agent_id].get_log_prob(obs_agent, actions_agent)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                probs = self.actors[agent_id](obs_agent)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                # Check for NaN gradients
                for param in self.actors[agent_id].parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                self.actor_optimizers[agent_id].step()
                
                total_actor_loss += actor_loss.item()
            
            # Critic update
            values = self.critic(global_obs).squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Check for NaN gradients
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()
        
        self.buffer.clear()
        
        return {
            "actor_loss": total_actor_loss / (n_epochs * self.n_agents),
            "critic_loss": total_critic_loss / n_epochs
        }
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros_like(rewards[0]) if len(rewards) > 0 else 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        
        return returns
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
        self.critic.load_state_dict(checkpoint['critic'])

