#!/usr/bin/env python
"""Test HRM Training with Fixed Vocabulary"""

import torch
import logging
from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel
from core.training.hrm_trainer import HRMTrainer
from core.auto_configure import AutoConfigurator

logging.basicConfig(level=logging.INFO)

print("="*70)
print("HRM TRAINING TEST - FIXED VOCABULARY")
print("="*70)

# Load config with new vocab_size
config = AutoConfigurator.load_config()

print(f"\n✓ Configuration loaded:")
print(f"  vocab_size: {config.vocab_size:,}")
print(f"  d_model: {config.d_model}")
print(f"  high_level_layers: {config.high_level_layers}")
print(f"  low_level_layers: {config.low_level_layers}")
print(f"  n_heads: {config.n_heads}")

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print(f"\n⚠ Using CPU (slower)")

print(f"\nInitializing HRM model...")

# Create HRM model
hrm = HierarchicalReasoningModel(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    high_level_layers=config.high_level_layers,
    low_level_layers=config.low_level_layers,
    n_heads=config.n_heads
)

# Count parameters
total_params = sum(p.numel() for p in hrm.parameters())
print(f"✓ HRM initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

# Initialize trainer
print(f"\nInitializing HRM trainer...")
trainer = HRMTrainer(
    hrm_model=hrm,
    device=device,
    learning_rate=1e-4
)

print(f"✓ Trainer initialized")
print(f"  Tokenizer vocab size: {len(trainer.tokenizer)}")
print(f"  HRM vocab size: {config.vocab_size}")

# Verify vocabulary match
if len(trainer.tokenizer) == config.vocab_size:
    print(f"\n✅ VOCABULARY SIZES MATCH!")
else:
    print(f"\n⚠ Warning: Vocabulary mismatch!")
    print(f"  Tokenizer: {len(trainer.tokenizer)}")
    print(f"  HRM: {config.vocab_size}")

# Load training data
print(f"\nLoading training data...")
try:
    examples = trainer.load_training_data("data/hrm_training_data.json")
    print(f"✓ Loaded {len(examples)} training examples")

    if len(examples) < 10:
        print(f"⚠ Warning: Only {len(examples)} examples (need at least 10 for testing)")

except FileNotFoundError:
    print(f"✗ Training data not found: data/hrm_training_data.json")
    print(f"  Run autonomous operation first to collect data")
    exit(1)

# Test forward pass with actual training data
print(f"\nTesting forward pass with real training data...")

try:
    # Prepare a small batch (3 examples)
    batch_examples = examples[:3]

    # Encode states using HRMTrainer's method
    input_ids_list = []
    batch_actions = []

    for example in batch_examples:
        # Encode state (returns already tokenized tensor)
        state = example['state']
        state_ids = trainer.encode_state(state)
        input_ids_list.append(state_ids)

        # Get action
        action_idx = trainer.encode_action(example['action'])
        batch_actions.append(action_idx)

    # Stack into batch
    input_ids = torch.stack(input_ids_list).to(device)
    attention_mask = None  # HRM doesn't use attention mask

    print(f"✓ Batch prepared:")
    print(f"  Batch size: {input_ids.shape[0]}")
    print(f"  Sequence length: {input_ids.shape[1]}")
    print(f"  Token ID range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"  HRM vocab size: {config.vocab_size}")

    # Check if token IDs are in valid range
    max_token = input_ids.max().item()
    if max_token >= config.vocab_size:
        print(f"\n✗ ERROR: Token ID {max_token} exceeds vocab size {config.vocab_size}!")
        exit(1)
    else:
        print(f"✓ All token IDs are within valid range [0, {config.vocab_size-1}]")

    # Move model to device
    hrm = hrm.to(device)
    hrm.eval()

    # Forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        outputs = hrm(input_ids, mask=None)
        logits = outputs['logits']

    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output: (batch={logits.shape[0]}, seq={logits.shape[1]}, vocab={logits.shape[2]})")

    # Check output dimensions
    assert logits.shape[-1] == config.vocab_size, "Output vocab size mismatch!"

    print(f"\n" + "="*70)
    print(f"✅ HRM TRAINING TEST PASSED!")
    print(f"="*70)
    print(f"\nHRM is ready to train:")
    print(f"  ✓ Vocabulary size: {config.vocab_size:,} tokens")
    print(f"  ✓ Model size: {total_params/1e6:.2f}M parameters")
    print(f"  ✓ Training data: {len(examples)} examples")
    print(f"  ✓ Device: {device}")
    print(f"  ✓ Forward pass: Working!")

    print(f"\nNext steps:")
    print(f"  1. Run: python start_training.py")
    print(f"  2. System will automatically switch from GPT-2 to HRM when ready")
    print(f"  3. HRM will continue improving through RL")

except Exception as e:
    print(f"\n✗ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
