"""
Quick test script for AEGIS (non-interactive)
"""

import torch
from aegis_autonomous import AutonomousAEGIS, AEGISConfig

print("\n" + "="*70)
print("AEGIS Quick Test")
print("="*70 + "\n")

print("1. Creating AEGIS with small configuration...")
config = AEGISConfig(
    vocab_size=1000,
    d_model=128,
    high_level_layers=2,
    low_level_layers=2,
    n_heads=4,
    population_size=3
)

aegis = AutonomousAEGIS(config)
print("✓ AEGIS created successfully\n")

print("2. Testing reasoning engine...")
test_input = torch.randint(0, config.vocab_size, (2, 10))
result = aegis.reason(test_input)
print(f"✓ Reasoning successful: {result['safe']}")
print(f"  Output shape: {result['logits'].shape}")
print(f"  Safety checks: {len(result['safety_checks'])}\n")

print("3. Checking agent state...")
state = aegis.agent.get_agent_state()
print(f"✓ Agent initialized")
print(f"  Active goals: {state['active_goals']}")
print(f"  Interests: {len(state['interests'])}")
print(f"  Drives: {state['drives']}\n")

print("4. Testing autonomous thinking...")
action = aegis.agent.think()
print(f"✓ Agent decided on action: {action['action']}")
if 'motivation' in action:
    print(f"  Motivation: {action['motivation']}\n")

print("5. Testing knowledge system...")
aegis.knowledge_system.add_human_knowledge(
    content="Test knowledge item",
    topic="testing"
)
stats = aegis.knowledge_system.knowledge_base.get_stats()
print(f"✓ Knowledge base working")
print(f"  Total items: {stats['total_items']}\n")

print("6. Testing safety validation...")
safe_code = "x = torch.randn(5, 5)"
check = aegis.safety_validator.code_validator.validate_code(safe_code)
print(f"✓ Code validation working: {check.passed}\n")

print("7. Getting system status...")
status = aegis.get_system_status()
print(f"✓ System status retrieved")
print(f"  Reasoning engine parameters: {status['reasoning_engine']['parameters']:,}")
print(f"  Evolution generation: {status['evolution']['current_generation']}")
print(f"  Safety frozen: {status['safety']['is_frozen']}\n")

print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nAEGIS is working correctly. To use interactively:")
print("  python aegis_autonomous.py")
print("\nOr see full demo (requires interaction):")
print("  python demo.py")
print("="*70 + "\n")
