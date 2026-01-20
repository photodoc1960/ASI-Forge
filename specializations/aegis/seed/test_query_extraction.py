#!/usr/bin/env python
"""
Quick test to see what GPT-2 is generating and how query extraction works
"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

# Enable DEBUG logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("="*70)
print("TESTING QUERY EXTRACTION")
print("="*70)

# Initialize AEGIS
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

print("\nRunning agent think() to observe GPT-2 generation...")
print("="*70)

# Agent thinks and decides action
action = aegis.agent.think()

print("\n" + "="*70)
print(f"Action decided: {action}")
print("\nCheck logs above for DEBUG messages showing:")
print("  1. What GPT-2 generated (raw_generation)")
print("  2. How query was extracted")
print("  3. Which fallback was used (if any)")
print("="*70)
