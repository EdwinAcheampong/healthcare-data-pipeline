"""
PPO (Proximal Policy Optimization) Agent for Healthcare Workload Management

This module implements a PPO agent with healthcare-specific compliance constraints
and safety mechanisms for optimal resource allocation in healthcare environments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    
    # Network architecture
    hidden_size: int = 256
    num_layers: int = 3
    
    # Training parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training iterations
    epochs_per_update: int = 10
    batch_size: int = 64
    
    # Healthcare-specific constraints
    compliance_threshold: float = 0.8
    safety_threshold: float = 0.9
    max_staff_change: int = 10
    max_bed_change: int = 20
    max_equipment_change: int = 5
    
    # Reward shaping
    compliance_penalty: float = -10.0
    safety_penalty: float = -20.0
    extreme_action_penalty: float = -5.0


class HealthcareActorCritic(nn.Module):
    """
    Actor-Critic network for healthcare workload management.
    
    The actor network outputs actions for resource allocation,
    while the critic network estimates state values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        
        self.config = config
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i in range(config.num_layers):
            self.feature_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
            input_dim = config.hidden_size
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, action_dim),
            nn.Tanh()  # Output actions in [-1, 1] range
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Action log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        logger.info(f"Actor-Critic network initialized with {state_dim} states and {action_dim} actions")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            action_mean: Mean of action distribution
            value: State value estimate
        """
        # Feature extraction
        features = state
        for layer in self.feature_layers:
            features = layer(features)
        
        # Actor and critic outputs
        action_mean = self.actor(features)
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for given state.
        
        Args:
            state: Input state tensor
            action: Optional action tensor (for evaluation)
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        action_mean, value = self.forward(state)
        
        # Create action distribution
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class HealthcareComplianceChecker:
    """
    Healthcare compliance checker for validating actions.
    
    This class ensures that RL actions comply with healthcare regulations
    and safety requirements.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        logger.info("Healthcare compliance checker initialized")
    
    def check_action_compliance(
        self, 
        action: torch.Tensor, 
        current_state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an action complies with healthcare regulations.
        
        Args:
            action: Action tensor from the agent
            current_state: Current environment state
            
        Returns:
            is_compliant: Whether the action is compliant
            compliance_info: Detailed compliance information
        """
        compliance_info = {
            'staff_changes_ok': True,
            'bed_changes_ok': True,
            'equipment_changes_ok': True,
            'safety_ok': True,
            'compliance_ok': True,
            'violations': []
        }
        
        # Parse action components
        add_staff = action[0].item()
        remove_staff = action[1].item()
        add_beds = action[2].item()
        remove_beds = action[3].item()
        add_equipment = action[4].item()
        remove_equipment = action[5].item()
        
        current_staff = current_state.get('current_staff', 0)
        current_beds = current_state.get('current_beds', 0)
        current_patients = current_state.get('current_patients', 0)
        
        # Check staff change limits
        net_staff_change = add_staff - remove_staff
        if abs(net_staff_change) > self.config.max_staff_change:
            compliance_info['staff_changes_ok'] = False
            compliance_info['violations'].append(f"Staff change {net_staff_change} exceeds limit {self.config.max_staff_change}")
        
        # Check bed change limits
        net_bed_change = add_beds - remove_beds
        if abs(net_bed_change) > self.config.max_bed_change:
            compliance_info['bed_changes_ok'] = False
            compliance_info['violations'].append(f"Bed change {net_bed_change} exceeds limit {self.config.max_bed_change}")
        
        # Check equipment change limits
        net_equipment_change = add_equipment - remove_equipment
        if abs(net_equipment_change) > self.config.max_equipment_change:
            compliance_info['equipment_changes_ok'] = False
            compliance_info['violations'].append(f"Equipment change {net_equipment_change} exceeds limit {self.config.max_equipment_change}")
        
        # Check minimum staffing requirements
        new_staff = current_staff + net_staff_change
        staff_patient_ratio = new_staff / max(current_patients, 1)
        if staff_patient_ratio < 0.05:  # Minimum 5% staff-to-patient ratio
            compliance_info['safety_ok'] = False
            compliance_info['violations'].append(f"Staff-to-patient ratio {staff_patient_ratio:.3f} below safety threshold 0.05")
        
        # Check minimum bed requirements
        new_beds = current_beds + net_bed_change
        if new_beds < current_patients * 0.8:  # At least 80% bed coverage
            compliance_info['safety_ok'] = False
            compliance_info['violations'].append(f"Bed coverage {new_beds/current_patients:.3f} below safety threshold 0.8")
        
        # Overall compliance
        compliance_info['compliance_ok'] = all([
            compliance_info['staff_changes_ok'],
            compliance_info['bed_changes_ok'],
            compliance_info['equipment_changes_ok'],
            compliance_info['safety_ok']
        ])
        
        return compliance_info['compliance_ok'], compliance_info
    
    def apply_compliance_constraints(
        self, 
        action: torch.Tensor, 
        current_state: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply compliance constraints to an action.
        
        Args:
            action: Original action tensor
            current_state: Current environment state
            
        Returns:
            constrained_action: Action with compliance constraints applied
        """
        constrained_action = action.clone()
        
        # Parse current state
        current_staff = current_state.get('current_staff', 0)
        current_beds = current_state.get('current_beds', 0)
        current_patients = current_state.get('current_patients', 0)
        
        # Apply staff constraints
        add_staff = action[0].item()
        remove_staff = action[1].item()
        net_staff_change = add_staff - remove_staff
        
        if abs(net_staff_change) > self.config.max_staff_change:
            # Scale down the change
            scale_factor = self.config.max_staff_change / abs(net_staff_change)
            constrained_action[0] = add_staff * scale_factor
            constrained_action[1] = remove_staff * scale_factor
        
        # Apply bed constraints
        add_beds = action[2].item()
        remove_beds = action[3].item()
        net_bed_change = add_beds - remove_beds
        
        if abs(net_bed_change) > self.config.max_bed_change:
            # Scale down the change
            scale_factor = self.config.max_bed_change / abs(net_bed_change)
            constrained_action[2] = add_beds * scale_factor
            constrained_action[3] = remove_beds * scale_factor
        
        # Apply equipment constraints
        add_equipment = action[4].item()
        remove_equipment = action[5].item()
        net_equipment_change = add_equipment - remove_equipment
        
        if abs(net_equipment_change) > self.config.max_equipment_change:
            # Scale down the change
            scale_factor = self.config.max_equipment_change / abs(net_equipment_change)
            constrained_action[4] = add_equipment * scale_factor
            constrained_action[5] = remove_equipment * scale_factor
        
        # Ensure minimum safety requirements
        new_staff = current_staff + (constrained_action[0] - constrained_action[1]).item()
        new_beds = current_beds + (constrained_action[2] - constrained_action[3]).item()
        
        # Adjust if minimum staffing not met
        min_required_staff = current_patients * 0.05
        if new_staff < min_required_staff:
            staff_deficit = min_required_staff - new_staff
            constrained_action[0] += staff_deficit
        
        # Adjust if minimum bed coverage not met
        min_required_beds = current_patients * 0.8
        if new_beds < min_required_beds:
            bed_deficit = min_required_beds - new_beds
            constrained_action[2] += bed_deficit
        
        return constrained_action


class PPOHHealthcareAgent:
    """
    PPO agent for healthcare workload management with compliance constraints.
    
    This agent implements Proximal Policy Optimization with healthcare-specific
    safety mechanisms and compliance checking.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        config: Optional[PPOConfig] = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration
        """
        self.config = config or PPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize networks
        self.actor_critic = HealthcareActorCritic(state_dim, action_dim, self.config)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.learning_rate)
        
        # Initialize compliance checker
        self.compliance_checker = HealthcareComplianceChecker(self.config)
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.dones = []
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'compliance_violations': [],
            'safety_violations': [],
            'value_losses': [],
            'policy_losses': []
        }
        
        logger.info(f"PPO Healthcare Agent initialized with {state_dim} states and {action_dim} actions")
    
    def select_action(
        self, 
        state: np.ndarray, 
        current_state_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select an action for the given state.
        
        Args:
            state: Current state observation
            current_state_dict: Additional state information for compliance checking
            
        Returns:
            action: Selected action
            action_info: Additional action information
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(state_tensor)
            
            # Apply compliance constraints if state dict is provided
            if current_state_dict is not None:
                # Check compliance
                is_compliant, compliance_info = self.compliance_checker.check_action_compliance(
                    action.squeeze(0), current_state_dict
                )
                
                # Apply constraints if not compliant
                if not is_compliant:
                    action = self.compliance_checker.apply_compliance_constraints(
                        action.squeeze(0), current_state_dict
                    ).unsqueeze(0)
                    logger.warning(f"Action constrained due to compliance violations: {compliance_info['violations']}")
                
                action_info = {
                    'log_prob': log_prob.item(),
                    'entropy': entropy.item(),
                    'value': value.item(),
                    'is_compliant': is_compliant,
                    'compliance_info': compliance_info
                }
            else:
                action_info = {
                    'log_prob': log_prob.item(),
                    'entropy': entropy.item(),
                    'value': value.item(),
                    'is_compliant': True,
                    'compliance_info': {'violations': []}
                }
            
            return action.squeeze(0).numpy(), action_info
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        value: float, 
        log_prob: float, 
        entropy: float, 
        done: bool
    ):
        """Store a transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.dones.append(done)
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            training_stats: Training statistics
        """
        if len(self.states) < self.config.batch_size:
            logger.warning("Not enough samples for policy update")
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Training statistics
        value_losses = []
        policy_losses = []
        
        # Multiple epochs of updates
        for epoch in range(self.config.epochs_per_update):
            # Create batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                action_mean, values = self.actor_critic(batch_states)
                action_std = torch.exp(self.actor_critic.log_std)
                dist = Normal(action_mean, action_std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # Compute ratios
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config.value_loss_coef * value_loss - 
                    self.config.entropy_coef * entropy.mean()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()
        self.dones.clear()
        
        training_stats = {
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
            'avg_value': np.mean(value_losses),
            'avg_policy': np.mean(policy_losses)
        }
        
        self.training_stats['value_losses'].extend(value_losses)
        self.training_stats['policy_losses'].extend(policy_losses)
        
        return training_stats
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from file."""
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return self.training_stats.copy()
    
    def log_episode_stats(self, episode_reward: float, episode_length: int, compliance_violations: int = 0, safety_violations: int = 0):
        """Log episode statistics."""
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['compliance_violations'].append(compliance_violations)
        self.training_stats['safety_violations'].append(safety_violations)
