"""
RL Trainer - Phase 3: Reinforcement Learning Fine-Tuning

Continuously improves the HRM through reinforcement learning based on
action outcomes. The HRM learns from experience and can surpass GPT-2
by optimizing directly for agent goals.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Deque
from collections import deque

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Reinforcement learning trainer for HRM.

    Uses policy gradient (REINFORCE algorithm) to improve HRM
    decision-making based on reward signals from action outcomes.
    """

    def __init__(
        self,
        hrm_model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        buffer_size: int = 1000
    ):
        """
        Initialize RL trainer.

        Args:
            hrm_model: The HRM model to fine-tune
            device: Device to train on
            learning_rate: Learning rate for policy gradient updates
            gamma: Discount factor for future rewards
            buffer_size: Size of experience replay buffer
        """
        self.model = hrm_model.to(device)
        self.device = device
        self.gamma = gamma

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Experience buffer for policy gradient
        self.experience_buffer: Deque = deque(maxlen=buffer_size)

        # Training statistics
        self.total_updates = 0
        self.cumulative_reward = 0.0
        self.episode_rewards: List[float] = []

        logger.info(f"RLTrainer initialized (device={device}, lr={learning_rate}, gamma={gamma})")

    def calculate_reward(self, action: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward for an action outcome.

        Args:
            action: Action that was taken
            result: Result of executing the action

        Returns:
            Reward value (positive for good outcomes, negative for bad)
        """
        reward = 0.0
        action_type = action.get('action', 'idle')

        # Web search rewards
        if action_type == 'web_search':
            knowledge_gained = result.get('knowledge_gained', 0)
            if knowledge_gained > 0:
                reward += 1.0 * knowledge_gained  # +1 per knowledge item
                logger.debug(f"Web search reward: +{reward:.2f} ({knowledge_gained} items)")
            else:
                reward -= 0.5  # Penalty for wasted search
                logger.debug("Web search penalty: -0.5 (no knowledge gained)")

        # Improvement proposal rewards
        elif action_type == 'propose_improvement':
            status = result.get('status', 'failed')

            if status == 'approved' or status == 'auto_approved':
                reward += 10.0  # Big reward for approved improvement!
                logger.debug("Improvement approved reward: +10.0")

            elif status == 'pending_approval':
                reward += 1.0  # Smaller reward for valid proposal
                logger.debug("Improvement pending reward: +1.0")

            elif status == 'already_pending':
                reward -= 2.0  # Penalty for duplicate
                logger.debug("Duplicate proposal penalty: -2.0")

            elif status == 'rejected':
                reward -= 1.0  # Penalty for rejected proposal
                logger.debug("Rejected proposal penalty: -1.0")

        # Ask human rewards
        elif action_type == 'ask_human':
            status = result.get('status', 'failed')
            if status == 'completed':
                reward += 0.5  # Small reward for getting answer
                logger.debug("Ask human reward: +0.5")
            else:
                reward -= 0.1  # Small penalty if failed
                logger.debug("Ask human penalty: -0.1")

        # Idle penalty
        elif action_type == 'idle':
            reward -= 0.1  # Small penalty for doing nothing
            logger.debug("Idle penalty: -0.1")

        return reward

    def store_experience(
        self,
        state: torch.Tensor,
        action_logits: torch.Tensor,
        action_idx: int,
        reward: float
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: State tensor
            action_logits: Model's action logits
            action_idx: Index of action taken
            reward: Reward received
        """
        experience = {
            'state': state.detach().cpu(),
            'action_logits': action_logits.detach().cpu(),
            'action_idx': action_idx,
            'reward': reward
        }

        self.experience_buffer.append(experience)
        self.cumulative_reward += reward

    def update_policy(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update policy using policy gradient (REINFORCE).

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary with update metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {'loss': 0.0, 'avg_reward': 0.0}

        self.model.train()

        # Sample batch from experience buffer
        import random
        batch = random.sample(self.experience_buffer, batch_size)

        total_loss = 0.0
        total_reward = 0.0

        self.optimizer.zero_grad()

        for experience in batch:
            state = experience['state'].to(self.device)
            action_logits = experience['action_logits'].to(self.device)
            action_idx = experience['action_idx']
            reward = experience['reward']

            # Compute log probability of action taken
            log_probs = F.log_softmax(action_logits, dim=-1)
            log_prob = log_probs[action_idx]

            # Policy gradient loss: -log(π(a|s)) * R
            # Negative because we want to maximize reward (minimize negative reward)
            loss = -log_prob * reward

            total_loss += loss
            total_reward += reward

        # Average loss over batch
        avg_loss = total_loss / batch_size

        # Backward pass
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.total_updates += 1

        metrics = {
            'loss': avg_loss.item(),
            'avg_reward': total_reward / batch_size,
            'buffer_size': len(self.experience_buffer),
            'total_updates': self.total_updates
        }

        logger.debug(f"Policy update: loss={metrics['loss']:.4f}, reward={metrics['avg_reward']:.2f}")

        return metrics

    def online_update(
        self,
        state: torch.Tensor,
        action_logits: torch.Tensor,
        action_idx: int,
        reward: float
    ) -> Dict[str, float]:
        """
        Perform immediate online update (single-step policy gradient).

        Args:
            state: State tensor
            action_logits: Model's action logits
            action_idx: Action taken
            reward: Reward received

        Returns:
            Update metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute log probability
        log_probs = F.log_softmax(action_logits, dim=-1)
        log_prob = log_probs[action_idx]

        # Policy gradient loss
        loss = -log_prob * reward

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.total_updates += 1
        self.cumulative_reward += reward

        return {
            'loss': loss.item(),
            'reward': reward,
            'total_updates': self.total_updates,
            'cumulative_reward': self.cumulative_reward
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_updates': self.total_updates,
            'cumulative_reward': self.cumulative_reward,
            'buffer_size': len(self.experience_buffer),
            'avg_reward': self.cumulative_reward / max(self.total_updates, 1)
        }

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save RL trainer checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'cumulative_reward': self.cumulative_reward,
            'buffer_size': len(self.experience_buffer)
        }

        torch.save(checkpoint, filepath)
        logger.info(f"RL checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load RL trainer checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.total_updates = checkpoint.get('total_updates', 0)
        self.cumulative_reward = checkpoint.get('cumulative_reward', 0.0)

        logger.info(f"RL checkpoint loaded: {filepath}")
