#!/usr/bin/env python
"""
Fast Data Collection for HRM Training

Runs autonomous agent to collect training data quickly.
Target: 100,000 examples for proper HRM training.
"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator
import time

logging.basicConfig(level=logging.INFO)

print("="*70)
print("FAST DATA COLLECTION FOR HRM TRAINING")
print("="*70)

# Load config
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

# Current status
current = len(aegis.data_collector.examples)
target = 1000000  # 1M examples for excellent training (29 params/example ratio)

print(f"\nCurrent data: {current:,} examples")
print(f"Target: {target:,} examples")
print(f"Remaining: {target - current:,} examples")
print()

if current >= target:
    print("✅ Already have enough data!")
    print(f"Ready to train HRM with {current:,} examples")
    print("\nRun: python train_hrm.py")
    exit(0)

# Estimate time
examples_per_iteration = 1  # Agent makes 1 decision per iteration
iterations_needed = target - current
seconds_per_iteration = 5  # Approximate
estimated_hours = (iterations_needed * seconds_per_iteration) / 3600

print(f"Estimated time to collect {target - current:,} examples:")
print(f"  At ~1 example per 5 seconds: {estimated_hours:.1f} hours")
print(f"  At ~1 example per 2 seconds: {estimated_hours * 2/5:.1f} hours (optimized)")
print()

print("Tips for faster collection:")
print("  1. Run multiple instances in parallel (separate terminals)")
print("  2. Each instance saves to same data file (automatically merged)")
print("  3. Let it run overnight!")
print()

response = input("Start data collection now? (y/n): ")
if response.lower() != 'y':
    print("Cancelled")
    exit(0)

print("\n" + "="*70)
print("STARTING DATA COLLECTION")
print("="*70)
print("\nPress Ctrl+C to stop and save progress")
print()

# Run autonomous loop with periodic saves
try:
    iteration = 0
    last_save = 0
    start_count = current

    while True:
        # Run one iteration
        aegis.run_iteration()
        iteration += 1

        # Save every 100 iterations
        if iteration - last_save >= 100:
            aegis.data_collector.save()
            new_count = len(aegis.data_collector.examples)
            collected_this_session = new_count - start_count

            print(f"\n{'='*70}")
            print(f"Progress Update (Iteration {iteration})")
            print(f"{'='*70}")
            print(f"  Total examples: {new_count:,}")
            print(f"  Collected this session: {collected_this_session:,}")
            print(f"  Target: {target:,}")
            print(f"  Remaining: {target - new_count:,}")
            print(f"  Progress: {100*new_count/target:.1f}%")

            if new_count >= target:
                print(f"\n✅ TARGET REACHED!")
                print(f"Collected {new_count:,} examples")
                print(f"\nReady to train HRM!")
                print(f"Run: python train_hrm.py")
                break

            last_save = iteration

        # Small delay
        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"\n\nStopping...")
    aegis.data_collector.save()
    final_count = len(aegis.data_collector.examples)
    print(f"\n{'='*70}")
    print(f"Data Collection Stopped")
    print(f"{'='*70}")
    print(f"  Total examples: {final_count:,}")
    print(f"  Collected this session: {final_count - start_count:,}")
    print(f"  Progress: {100*final_count/target:.1f}%")
    print(f"\nData saved to: data/hrm_training_data.json")

    if final_count >= target:
        print(f"\n✅ Ready to train!")
        print(f"Run: python train_hrm.py")
    else:
        print(f"\n⏳ Need {target - final_count:,} more examples")
        print(f"Run this script again to continue collecting")
