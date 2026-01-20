"""
HRM Trainer - Phase 2: Imitation Learning

Trains the Hierarchical Reasoning Model to imitate GPT-2's decision-making
using data collected during autonomous operation.
"""

import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


class HRMTrainer:
    """
    Trains HRM through supervised imitation learning.

    The HRM learns to predict the same actions that GPT-2 would make,
    given the same agent state. This allows HRM to distill GPT-2's
    decision-making into a much smaller model.
    """

    def __init__(
        self,
        hrm_model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        Initialize HRM trainer.

        Args:
            hrm_model: The HRM model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = hrm_model.to(device)
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        # Action vocabulary mapping
        self.action_to_idx = {
            'web_search': 0,
            'ask_human': 1,
            'propose_improvement': 2,
            'idle': 3
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}

        logger.info(f"HRMTrainer initialized (device={device}, lr={learning_rate})")

    def load_training_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load training data collected from GPT-2.

        Args:
            filepath: Path to training data JSON file

        Returns:
            List of training examples
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        examples = data.get('examples', [])
        logger.info(f"Loaded {len(examples)} training examples from {filepath}")

        return examples

    def encode_state(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dictionary to tensor input for HRM.

        Args:
            state_dict: Dictionary containing agent state

        Returns:
            Encoded state tensor
        """
        # SIMPLIFIED: Just encode action types as simple text
        # This avoids complex tokenization issues

        # Build simple state description
        state_text = "Agent state: "

        # Add goal count
        goal_count = state_dict.get('goal_count', 0)
        state_text += f"{goal_count} goals. "

        # Add curiosity count
        curiosity_count = state_dict.get('curiosity_count', 0)
        state_text += f"{curiosity_count} curiosities. "

        # Add knowledge count
        knowledge_count = state_dict.get('knowledge_count', 0)
        state_text += f"{knowledge_count} knowledge items."

        # Tokenize (shorter text = less likely to cause issues)
        tokens = self.tokenizer(
            state_text,
            max_length=64,  # Much shorter
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return tokens['input_ids'].squeeze(0).to(self.device)

    def encode_action(self, action_dict: Dict[str, Any]) -> int:
        """
        Convert action dictionary to class index.

        Args:
            action_dict: Action taken by GPT-2

        Returns:
            Integer class index (0-3)
        """
        action_type = action_dict.get('action', 'idle')
        action_idx = self.action_to_idx.get(action_type, 3)  # Default to 'idle'

        # Ensure valid range [0, 3]
        if action_idx < 0 or action_idx >= len(self.action_to_idx):
            logger.warning(f"Invalid action index {action_idx} for action {action_type}, using 'idle'")
            action_idx = 3

        return action_idx

    def train_epoch(self, examples: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            examples: List of training examples

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0

        for i, example in enumerate(examples):
            try:
                # Encode state
                state = self.encode_state(example['state'])

                # Encode target action
                action_idx = self.encode_action(example['action'])

                # Validate action index
                if action_idx < 0 or action_idx >= len(self.action_to_idx):
                    logger.warning(f"Skipping example {i}: invalid action index {action_idx}")
                    continue

                target = torch.tensor([action_idx], dtype=torch.long).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # HRM expects input_ids as input
                try:
                    output = self.model(state.unsqueeze(0))  # Add batch dimension
                except Exception as e:
                    logger.error(f"Model forward failed on example {i}: {e}")
                    logger.error(f"  State shape: {state.shape}")
                    continue

                # Get action logits
                # HRM outputs a dictionary with 'logits' key (from GPT-2)
                if isinstance(output, dict):
                    if 'logits' in output:
                        # Average pool over sequence length to get per-example logits
                        try:
                            logits = output['logits'].mean(dim=1)  # [batch, vocab_size]
                        except Exception as e:
                            logger.error(f"Failed to pool logits on example {i}: {e}")
                            logger.error(f"  Logits shape: {output['logits'].shape}")
                            continue

                        # Project to action space (4 actions)
                        if logits.size(-1) != len(self.action_to_idx):
                            # Need a projection layer - use simple linear projection
                            if not hasattr(self, 'action_head'):
                                logger.info(f"Creating action head: {logits.size(-1)} -> {len(self.action_to_idx)}")
                                self.action_head = nn.Linear(
                                    logits.size(-1),
                                    len(self.action_to_idx)
                                ).to(self.device)

                            try:
                                logits = self.action_head(logits)
                            except Exception as e:
                                logger.error(f"Action head projection failed on example {i}: {e}")
                                logger.error(f"  Input shape: {logits.shape}")
                                continue
                    else:
                        logger.warning(f"HRM output missing 'logits' key: {output.keys()}")
                        continue
                else:
                    logits = output

                # Final validation before loss
                if logits.size(-1) != len(self.action_to_idx):
                    logger.warning(f"Logits size {logits.size(-1)} != num_actions {len(self.action_to_idx)}")
                    continue

                # Debug: Check target range
                if target.item() < 0 or target.item() >= len(self.action_to_idx):
                    logger.error(f"Example {i}: Target {target.item()} out of range [0, {len(self.action_to_idx)-1}]")
                    logger.error(f"  Action was: {example['action']}")
                    continue

                # Debug: Check logits shape
                if logits.shape != (1, len(self.action_to_idx)):
                    logger.error(f"Example {i}: Wrong logits shape {logits.shape}, expected (1, {len(self.action_to_idx)})")
                    continue

                # Compute loss
                try:
                    loss = self.criterion(logits, target)
                except RuntimeError as e:
                    if 'CUDA' in str(e):
                        logger.error(f"CUDA error at example {i}, clearing cache and skipping")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                predicted = logits.argmax(dim=-1)
                correct += (predicted == target).sum().item()

            except RuntimeError as e:
                if 'CUDA' in str(e):
                    logger.error(f"CUDA error at example {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                continue

        if len(examples) == 0:
            return 0.0, 0.0

        avg_loss = total_loss / len(examples)
        accuracy = correct / len(examples)

        return avg_loss, accuracy

    def train(
        self,
        data_file: str,
        epochs: int = 20,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            data_file: Path to training data JSON
            epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with training metrics history
        """
        # Load data
        examples = self.load_training_data(data_file)

        if len(examples) == 0:
            logger.warning("No training examples found!")
            return {'loss': [], 'accuracy': []}

        print(f"\n{'='*70}")
        print(f"HRM TRAINING - Phase 2: Imitation Learning")
        print(f"{'='*70}")
        print(f"Training on {len(examples)} examples for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*70}\n")

        # Training history
        history = {
            'loss': [],
            'accuracy': []
        }

        # Training loop
        for epoch in range(epochs):
            loss, acc = self.train_epoch(examples)
            history['loss'].append(loss)
            history['accuracy'].append(acc)

            print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, Accuracy={acc:.2%}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(checkpoint_dir) / f"hrm_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_path = Path(checkpoint_dir) / "hrm_trained.pt"
        self.save_checkpoint(final_path)
        print(f"\n✓ Training complete! Final model saved: {final_path}")
        print(f"  Final accuracy: {history['accuracy'][-1]:.2%}")

        return history

    def save_checkpoint(self, filepath: Path) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'action_vocab': self.action_to_idx
        }

        # Save action head if it exists
        if hasattr(self, 'action_head'):
            checkpoint['action_head_state'] = self.action_head.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load action head if present
        if 'action_head_state' in checkpoint:
            vocab_size = list(checkpoint['model_state'].values())[0].size(-1)
            self.action_head = nn.Linear(vocab_size, len(self.action_to_idx)).to(self.device)
            self.action_head.load_state_dict(checkpoint['action_head_state'])

        logger.info(f"Checkpoint loaded: {filepath}")

    def evaluate(self, data_file: str) -> Dict[str, float]:
        """
        Evaluate HRM on validation data.

        Args:
            data_file: Path to validation data JSON

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        examples = self.load_training_data(data_file)

        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for example in examples:
                try:
                    state = self.encode_state(example['state'])
                    target = torch.tensor([self.encode_action(example['action'])]).to(self.device)

                    output = self.model(state.unsqueeze(0))

                    if isinstance(output, dict) and 'logits' in output:
                        logits = output['logits'].mean(dim=1)
                        if hasattr(self, 'action_head'):
                            logits = self.action_head(logits)
                    else:
                        continue

                    loss = self.criterion(logits, target)
                    total_loss += loss.item()

                    predicted = logits.argmax(dim=-1)
                    correct += (predicted == target).sum().item()

                except Exception as e:
                    logger.warning(f"Error evaluating example: {e}")
                    continue

        return {
            'loss': total_loss / len(examples) if examples else 0.0,
            'accuracy': correct / len(examples) if examples else 0.0
        }
