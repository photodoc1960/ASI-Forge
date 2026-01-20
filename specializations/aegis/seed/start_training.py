#!/usr/bin/env python
"""
Start HRM Training - Direct Entry Point

Skips all demos and goes straight to autonomous operation for training data collection.
"""

import sys
import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("="*70)
print("HRM AUTONOMOUS TRAINING")
print("="*70)
print("\nThis will:")
print("  1. Collect training data from GPT-2 decisions (1,000,000 examples)")
print("  2. Automatically train HRM when ready")
print("  3. Switch to HRM and continue with RL fine-tuning")
print("\n  This will take ~6-10 days of continuous running")
print("\nPress Ctrl+C at any time to enter interactive mode")
print("="*70)

# Initialize AEGIS
print("\nInitializing AEGIS...")
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

print("\n✓ AEGIS initialized and ready!")
print(f"  Training Phase: {aegis.training_phase} (Data Collection)")
print(f"  Using: GPT-2 (124M parameters)")
print(f"  Current: {len(aegis.data_collector.examples):,} examples")
print(f"  Target: 1,000,000 examples")
print(f"  Remaining: {1000000 - len(aegis.data_collector.examples):,} examples")
print(f"  Estimated: ~{(1000000 - len(aegis.data_collector.examples)) / (2 * 3600):.1f} hours at 2/sec")
print()

# Start autonomous operation
try:
    aegis.start_autonomous_operation(
        max_iterations=10000000,  # Effectively unlimited (will stop at 1M examples)
        think_interval_seconds=0.5  # Faster: 0.5 seconds between iterations
    )
except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("PAUSED - Entering Interactive Mode")
    print("="*70)

    # Show current progress
    collector_stats = aegis.data_collector.get_stats()
    print(f"\nProgress:")
    print(f"  Examples collected: {collector_stats['total_examples']:,}/1,000,000")
    print(f"  Actions taken:")
    for action, count in collector_stats.get('action_distribution', {}).items():
        print(f"    {action}: {count}")

    print("\nAvailable commands:")
    print("  training  - Show detailed training status")
    print("  status    - Show overall system status")
    print("  start     - Resume autonomous operation")
    print("  quit      - Exit")
    print()

    # Enter interactive mode
    aegis.interactive_session()
