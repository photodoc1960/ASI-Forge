#!/usr/bin/env python
"""Start HRM Training Immediately"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

logging.basicConfig(level=logging.INFO)

print("="*70)
print("STARTING HRM TRAINING")
print("="*70)

# Load config
config = AutoConfigurator.load_config()

# Initialize AEGIS
aegis = AutonomousAEGIS(config)

# Check training data
num_examples = len(aegis.data_collector.examples)
print(f"\n✓ Training data: {num_examples:,} examples")

# Calculate parameters
from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel
test_hrm = HierarchicalReasoningModel(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    high_level_layers=config.high_level_layers,
    low_level_layers=config.low_level_layers,
    n_heads=config.n_heads
)
total_params = sum(p.numel() for p in test_hrm.parameters())
params_per_example = total_params / num_examples if num_examples > 0 else float('inf')

print(f"✓ HRM size: {total_params:,} parameters ({total_params/1e6:.1f}M)")
print(f"✓ Ratio: {params_per_example:,.0f} parameters per example")
print()

# Assess data sufficiency
if params_per_example > 1000:
    print(f"❌ SEVERE UNDERFITTING RISK!")
    print(f"   {params_per_example:,.0f} params/example is WAY too high")
    print(f"   Model will overfit and fail to generalize")
    print()
    print(f"Recommendations:")
    print(f"  1. Collect more data: Need {total_params/10:,.0f}+ examples (10 per param)")
    print(f"  2. Or collect {total_params/100:,.0f}+ examples minimum")
    print(f"  3. Currently only have {100*num_examples/(total_params/10):.1f}% of recommended data")
    print()
    print(f"Quick fix: Run 'python collect_data_fast.py' to collect 100K examples")
    print()
    response = input("Train anyway with high overfitting risk? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        print("\nCollect more data first:")
        print("  python collect_data_fast.py")
        exit(0)
elif params_per_example > 100:
    print(f"⚠ Warning: Data is on the low side")
    print(f"  {params_per_example:.0f} params/example")
    print(f"  Recommended: {total_params/10:,.0f}+ examples")
    print(f"  Current: {num_examples:,} examples ({100*num_examples/(total_params/10):.1f}% of recommended)")
    print()
    response = input("Continue with limited data? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        exit(0)
elif params_per_example > 10:
    print(f"✓ Acceptable data amount")
    print(f"  {params_per_example:.0f} params/example")
    print(f"  Should train reasonably well")
else:
    print(f"✅ Excellent data amount!")
    print(f"  {params_per_example:.0f} params/example")
    print(f"  Model should train very well")

# Start HRM training
print(f"\nStarting HRM training...")
print(f"This will:")
print(f"  1. Initialize HRM (29M parameters)")
print(f"  2. Train HRM to imitate GPT-2's decisions")
print(f"  3. Switch to HRM if accuracy ≥ 70%")
print(f"  4. Continue improving with RL")
print()

aegis._start_hrm_training()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

if aegis.using_hrm:
    print("\n✅ System now using HRM!")
    print(f"  HRM replaced GPT-2 as the reasoning engine")
    print(f"  Phase 3 (RL fine-tuning) is now active")
else:
    print("\n⚠ Still using GPT-2")
    print(f"  HRM accuracy was below 70% threshold")
    print(f"  Need more training data or more epochs")
