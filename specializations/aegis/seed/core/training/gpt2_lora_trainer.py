"""
GPT-2 LoRA Trainer - Fine-tune GPT-2 on Agent Decisions

Uses LoRA (Low-Rank Adaptation) to efficiently fine-tune GPT-2
on the agent's collected decision data.
"""

import json
import logging
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# Suppress loss_type warnings from transformers library
warnings.filterwarnings('ignore', message='.*loss_type.*')

logger = logging.getLogger(__name__)


class AgentDecisionDataset(Dataset):
    """Dataset of agent state -> action decisions"""

    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def _format_example(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convert example to (prompt, completion) format.

        Prompt: Agent state description
        Completion: Action to take
        """
        state = example['state']
        action = example['action']

        # Build prompt from state
        prompt_parts = ["Agent state:"]

        # Add goals
        goal_count = state.get('goal_count', 0)
        if goal_count > 0:
            prompt_parts.append(f"{goal_count} active goals.")
            if state.get('active_goals'):
                prompt_parts.append(f"Current goal: {state['active_goals'][0]}")

        # Add curiosity
        curiosity_count = state.get('curiosity_count', 0)
        if curiosity_count > 0:
            prompt_parts.append(f"{curiosity_count} knowledge gaps.")

        # Add knowledge
        knowledge_count = state.get('knowledge_count', 0)
        prompt_parts.append(f"{knowledge_count} knowledge items.")

        # Add recent actions
        recent = state.get('recent_actions', [])
        if recent:
            prompt_parts.append(f"Recent: {' -> '.join(recent[-3:])}")

        prompt = " ".join(prompt_parts)
        prompt += "\nBest action:"

        # Build completion from action
        action_type = action.get('action', 'idle')

        if action_type == 'web_search':
            query = action.get('query', 'relevant information')
            completion = f" web_search({query})"
        elif action_type == 'ask_human':
            question = action.get('question', 'guidance')
            completion = f" ask_human({question})"
        elif action_type == 'propose_improvement':
            aspect = action.get('aspect', 'architecture')
            completion = f" propose_improvement({aspect})"
        else:
            completion = f" {action_type}"

        return prompt, completion

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt, completion = self._format_example(example)

        # Combine prompt + completion for training
        full_text = prompt + completion

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels: -100 for prompt tokens (don't compute loss), actual tokens for completion
        labels = input_ids.clone()

        # Find where completion starts (after "Best action:")
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True
        )
        prompt_length = len(prompt_encoding['input_ids'])

        # Mask out prompt tokens in labels
        labels[:prompt_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class GPT2LoRATrainer:
    """
    Fine-tunes GPT-2 with LoRA on agent decision data.

    Uses PEFT library for efficient LoRA training.
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        device: str = 'cpu',
        learning_rate: float = 5e-5
    ):
        """
        Initialize GPT-2 LoRA trainer.

        Args:
            model: GPT-2 model to fine-tune
            tokenizer: GPT-2 tokenizer
            device: Device for training
            learning_rate: Learning rate for LoRA parameters
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate

        # Apply LoRA
        self.peft_model = self._apply_lora(model)

        # Move to device
        self.peft_model.to(device)

        # Setup optimizer (only train LoRA parameters)
        self.optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=learning_rate
        )

        logger.info(f"GPT2LoRATrainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {learning_rate}")

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

    def _apply_lora(self, model):
        """Apply LoRA adapters to GPT-2 model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Fix model config to avoid loss_type warning
            if hasattr(model.config, 'loss_type') and model.config.loss_type is None:
                delattr(model.config, 'loss_type')

            lora_config = LoraConfig(
                r=8,  # rank
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            peft_model = get_peft_model(model, lora_config)

            # Thoroughly clean loss_type from all config locations
            # PEFT creates nested model structures, clean them all
            configs_to_clean = [
                peft_model.config if hasattr(peft_model, 'config') else None,
                peft_model.base_model.config if hasattr(peft_model, 'base_model') and hasattr(peft_model.base_model, 'config') else None,
                peft_model.base_model.model.config if hasattr(peft_model, 'base_model') and hasattr(peft_model.base_model, 'model') and hasattr(peft_model.base_model.model, 'config') else None,
            ]

            for cfg in configs_to_clean:
                if cfg is not None and hasattr(cfg, 'loss_type') and cfg.loss_type is None:
                    delattr(cfg, 'loss_type')

            logger.info("✓ LoRA adapters applied to GPT-2")

            return peft_model

        except ImportError:
            logger.error("PEFT library not available - cannot apply LoRA")
            logger.error("Install with: pip install peft")
            raise

    def load_training_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Load collected training data"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        examples = data.get('examples', [])
        logger.info(f"Loaded {len(examples)} training examples from {filepath}")

        return examples

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, int]:
        """Train for one epoch"""
        self.peft_model.train()
        total_loss = 0.0
        num_batches = 0

        # Suppress loss_type warning during training
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*loss_type.*')

            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                outputs = self.peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss, num_batches

    def train(
        self,
        data_file: str,
        epochs: int = 3,
        batch_size: int = 4,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            data_file: Path to training data JSON
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with training metrics
        """
        # Load data
        examples = self.load_training_data(data_file)

        if len(examples) == 0:
            logger.warning("No training examples found!")
            return {'loss': []}

        # Create dataset and dataloader
        dataset = AgentDecisionDataset(examples, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print(f"\n{'='*70}")
        print(f"GPT-2 + LoRA FINE-TUNING")
        print(f"{'='*70}")
        print(f"Training on {len(examples)} examples for {epochs} epochs")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        # Training history
        history = {'loss': []}

        # Training loop
        for epoch in range(epochs):
            loss, num_batches = self.train_epoch(dataloader)
            history['loss'].append(loss)

            print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f} ({num_batches} batches)")

            # Save checkpoint every epoch
            checkpoint_path = Path(checkpoint_dir) / f"gpt2_lora_epoch_{epoch+1}.pt"
            self.save_checkpoint(checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_path = Path(checkpoint_dir) / "gpt2_lora_trained.pt"
        self.save_checkpoint(final_path)

        print(f"\n✓ Training complete! Final model saved: {final_path}")
        print(f"  Final loss: {history['loss'][-1]:.4f}")

        return history

    def save_checkpoint(self, filepath: Path) -> None:
        """Save LoRA adapter weights"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter state
        self.peft_model.save_pretrained(filepath.parent / filepath.stem)

        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """Load LoRA adapter weights"""
        from peft import PeftModel

        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            filepath.parent / filepath.stem
        )

        self.peft_model.to(self.device)

        logger.info(f"Checkpoint loaded: {filepath}")
