"""
Training Infrastructure

Implements autonomous training:
1. Data Collection: Collect GPT-2 decisions during autonomous operation
2. GPT-2 LoRA Fine-tuning: Improve GPT-2 with LoRA on collected data
3. RL Fine-tuning: Further improve through reinforcement learning
"""

from .data_collector import TrainingDataCollector
from .gpt2_lora_trainer import GPT2LoRATrainer
from .rl_trainer import RLTrainer
from .hrm_trainer import HRMTrainer

__all__ = ['TrainingDataCollector', 'GPT2LoRATrainer', 'RLTrainer', 'HRMTrainer']
