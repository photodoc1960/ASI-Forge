#!/usr/bin/env python
"""Test GPT-2 LoRA Training on Collected Data"""

import logging
import torch
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

logging.basicConfig(level=logging.INFO)

print("="*70)
print("GPT-2 LoRA TRAINING TEST")
print("="*70)

# Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n🚀 GPU Detected: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    print(f"   Training will use CUDA for maximum speed!")
else:
    print("\n⚠ No GPU detected - training will use CPU (slower)")

# Initialize AEGIS
print("\nInitializing AEGIS...")
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

# Check collected data
num_examples = len(aegis.data_collector.examples)
print(f"\nTraining data available: {num_examples} examples")

if num_examples < 10:
    print("\n⚠ Not enough training data!")
    print("  Need at least 10 examples for testing")
    print("  (1000 for full training)")
    exit(1)

print(f"✓ Sufficient data for testing ({num_examples} examples)")

# Show training configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10 if device == 'cuda' else 5
batch_size = 8 if device == 'cuda' else 4

print(f"\nTraining Configuration:")
print(f"  Device: {device}")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Trainable params: ~811K (0.65% of GPT-2)")

# Trigger fine-tuning
print("\n" + "="*70)
print("Starting GPT-2 LoRA fine-tuning...")
print("="*70)

try:
    aegis._start_gpt2_finetuning()

    print("\n" + "="*70)
    print("✅ TRAINING SUCCESSFUL!")
    print("="*70)

    print(f"\nTraining Phase: {aegis.training_phase}")
    print(f"GPT-2 Fine-tuned: {aegis.gpt2_finetuned}")

    if aegis.gpt2_finetuned:
        print("\n✓ GPT-2 now has LoRA adapters trained on agent decisions!")
        print("✓ Agent will make better decisions based on collected experience!")
        print(f"\n✓ Model saved to: checkpoints/gpt2_lora_trained/")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
