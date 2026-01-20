"""
Test unsupervised self-training with auto-approval

This demonstrates AEGIS autonomously evolving its architecture
with low-risk improvements auto-approved.
"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.evolution.supervised_evolution import EvolutionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("Testing AEGIS Unsupervised Self-Training")
print("=" * 70)

# Create custom evolution config with auto-approval enabled
evolution_config = EvolutionConfig(
    population_size=5,
    max_generations=3,

    # Auto-approval settings
    enable_auto_approval=True,
    auto_approve_low_risk=True,
    min_performance_gain=0.02,  # 2% improvement
    max_auto_approve_param_increase=0.05,  # 5% param increase max

    # Still require approval for risky changes
    require_approval_for_deployment=True,
)

print("\nAuto-Approval Configuration:")
print(f"  Enabled: {evolution_config.enable_auto_approval}")
print(f"  Min performance gain: {evolution_config.min_performance_gain:.1%}")
print(f"  Max param increase: {evolution_config.max_auto_approve_param_increase:.1%}")

# Initialize AEGIS with pretrained LLM
print("\nInitializing AEGIS...")
aegis = AutonomousAEGIS(use_pretrained_llm=True)

# Replace evolution config
aegis.evolution_framework.config = evolution_config

print("\n" + "=" * 70)
print("Starting Evolution (3 generations)")
print("Low-risk improvements will be auto-approved")
print("=" * 70)

# Simple evaluation function
def eval_fn(model):
    """Dummy evaluation - returns random performance"""
    import random
    return {
        'accuracy': 0.75 + random.random() * 0.1,
        'loss': 2.0 - random.random() * 0.5
    }

# Run evolution
for gen in range(3):
    print(f"\n{'='*70}")
    print(f"Generation {gen + 1}")
    print('='*70)

    stats = aegis.evolution_framework.evolve_generation(eval_fn)

    print(f"\nGeneration {gen + 1} Stats:")
    print(f"  Status: {stats.get('status', 'completed')}")
    print(f"  New candidates: {stats.get('new_candidates', 0)}")
    print(f"  Population size: {stats.get('population_size', 0)}")

    if stats.get('best_performance'):
        print(f"  Best performance: {stats['best_performance']}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)

print("\nSummary:")
print(f"  Total generations: {aegis.evolution_framework.current_generation}")
print(f"  Population size: {len(aegis.evolution_framework.population)}")
print(f"  Archive size: {len(aegis.evolution_framework.archive)}")

# Count auto-approved vs human-approved
auto_approved = sum(1 for c in aegis.evolution_framework.archive if c.human_approved)
total = len(aegis.evolution_framework.archive)
print(f"  Auto-approved: {auto_approved}/{total}")
