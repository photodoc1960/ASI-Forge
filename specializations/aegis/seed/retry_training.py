#!/usr/bin/env python
"""Retry Phase 2 training with collected data"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

logging.basicConfig(level=logging.INFO)

print("="*70)
print("RETRYING PHASE 2: HRM TRAINING")
print("="*70)

config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

print(f"\nCurrent training phase: {aegis.training_phase}")
print(f"Training examples collected: {len(aegis.data_collector.examples)}")

print("\nStarting Phase 2 training...")
aegis._start_imitation_learning()
