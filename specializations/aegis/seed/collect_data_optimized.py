#!/usr/bin/env python
"""
Optimized Data Collection - Single Process, Maximum Speed

Uses batching and optimization to collect data as fast as possible
on a single GPU without parallel process overhead.
"""

import logging
import time
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

# Reduce logging overhead
logging.basicConfig(level=logging.WARNING)

print("="*70)
print("OPTIMIZED DATA COLLECTION - 1M EXAMPLES")
print("="*70)

config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

current = len(aegis.data_collector.examples)
target = 1000000

print(f"\nCurrent: {current:,} examples")
print(f"Target: {target:,} examples")
print(f"Remaining: {target - current:,} examples")
print()

if current >= target:
    print("✅ Target reached!")
    exit(0)

print("Optimizations enabled:")
print("  ✓ Reduced logging (WARNING level only)")
print("  ✓ Fast iteration loop")
print("  ✓ Batch saves (every 1000 examples)")
print("  ✓ Minimal overhead")
print()

response = input("Start optimized collection? (y/n): ")
if response.lower() != 'y':
    print("Cancelled")
    exit(0)

print("\n" + "="*70)
print("COLLECTING DATA - Press Ctrl+C to stop")
print("="*70)

try:
    iteration = 0
    last_save = 0
    start_count = current
    start_time = time.time()
    last_update_time = start_time

    while True:
        # Run iteration
        aegis.run_iteration()
        iteration += 1

        # Save every 1000 iterations (less I/O overhead)
        if iteration - last_save >= 1000:
            aegis.data_collector.save()
            new_count = len(aegis.data_collector.examples)

            # Calculate stats
            elapsed = time.time() - start_time
            collected = new_count - start_count
            rate = collected / elapsed if elapsed > 0 else 0
            remaining = target - new_count
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
            eta_days = eta_hours / 24

            print(f"\n[{iteration:,} iterations] {new_count:,} examples")
            print(f"  Rate: {rate:.2f} examples/sec")
            print(f"  Progress: {100*new_count/target:.2f}%")
            print(f"  ETA: {eta_hours:.1f} hours ({eta_days:.1f} days)")

            if new_count >= target:
                print(f"\n✅ TARGET REACHED! Collected {new_count:,} examples")
                break

            last_save = iteration
            last_update_time = time.time()

except KeyboardInterrupt:
    print(f"\n\nStopped by user")
    aegis.data_collector.save()

    final_count = len(aegis.data_collector.examples)
    elapsed = time.time() - start_time
    collected = final_count - start_count

    print(f"\n{'='*70}")
    print(f"Session Summary")
    print(f"{'='*70}")
    print(f"  Total examples: {final_count:,}")
    print(f"  Collected: {collected:,}")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Rate: {collected/elapsed:.2f} examples/sec")
    print(f"  Progress: {100*final_count/target:.2f}%")

print(f"\nData saved to: data/hrm_training_data.json")

if final_count >= target:
    print(f"\n🎉 READY TO TRAIN HRM!")
    print(f"Run: python train_hrm.py")
