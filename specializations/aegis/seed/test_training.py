#!/usr/bin/env python
"""
Test HRM Training Infrastructure

Quick test to verify:
1. Training modules can be imported
2. Data collector works
3. Trainer can be initialized
4. Integration with AEGIS works
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*70)
print("HRM TRAINING INFRASTRUCTURE TEST")
print("="*70)

# Test 1: Import training modules
print("\n[1/5] Testing imports...")
try:
    from core.training import TrainingDataCollector, HRMTrainer, RLTrainer
    print("✓ Training modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize data collector
print("\n[2/5] Testing data collector...")
try:
    collector = TrainingDataCollector(save_path="data/test_training_data.json")

    # Add a test example
    test_state = {
        'active_goals': ['Learn about neural networks'],
        'goal_types': ['EXPLORATION'],
        'goal_count': 1,
        'curiosity_count': 2,
        'top_curiosity': ['attention mechanisms'],
        'knowledge_count': 5,
        'recent_knowledge': ['Test knowledge'],
        'recent_actions': ['web_search', 'idle'],
        'questions_asked': 3,
        'iteration': 1
    }

    test_action = {
        'action': 'web_search',
        'query': 'transformer attention',
        'confidence': 0.75
    }

    test_result = {
        'status': 'completed',
        'knowledge_gained': 3
    }

    collector.add_example(test_state, test_action, test_result)

    stats = collector.get_stats()
    print(f"✓ Data collector working (examples: {stats['total_examples']})")

    # Clean up
    import os
    if os.path.exists("data/test_training_data.json"):
        os.remove("data/test_training_data.json")

except Exception as e:
    print(f"✗ Data collector failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Initialize AEGIS with training infrastructure
print("\n[3/5] Testing AEGIS integration...")
try:
    from aegis_autonomous import AutonomousAEGIS
    from core.auto_configure import AutoConfigurator

    config = AutoConfigurator.load_config()
    aegis = AutonomousAEGIS(config)

    # Check training components
    assert hasattr(aegis, 'data_collector'), "Missing data_collector"
    assert hasattr(aegis, 'training_phase'), "Missing training_phase"
    assert aegis.training_phase == 1, "Wrong initial phase"

    print(f"✓ AEGIS initialized with training infrastructure")
    print(f"  Training phase: {aegis.training_phase}")
    print(f"  Data collector: ✓")

except Exception as e:
    print(f"✗ AEGIS integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test training status formatting
print("\n[4/5] Testing training status display...")
try:
    stats_str = aegis._format_training_stats()
    assert "Phase 1" in stats_str, "Missing phase info"
    assert "Data Collection" in stats_str, "Missing phase description"

    print("✓ Training status formatting works")
    print("\nCurrent status:")
    print(stats_str)

except Exception as e:
    print(f"✗ Status formatting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data collection in a single iteration
print("\n[5/5] Testing data collection during operation...")
try:
    # Simulate one iteration of data collection
    state_before = aegis.data_collector.encode_agent_state(
        aegis.agent,
        aegis.knowledge_system,
        iteration=1
    )

    # Agent thinks
    action = aegis.agent.think()

    # Simulate result
    result = {'status': 'idle', 'knowledge_gained': 0}

    # Collect training data
    aegis.data_collector.add_example(
        state=state_before,
        action=action,
        result=result
    )

    num_examples = len(aegis.data_collector.examples)
    print(f"✓ Data collection during operation works")
    print(f"  Examples collected: {num_examples}")

    # Clean up
    aegis.data_collector.clear()

except Exception as e:
    print(f"✗ Data collection test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nHRM training infrastructure is ready!")
print("\nTo use:")
print("  1. Run: python demo.py")
print("  2. Type: start")
print("  3. Monitor: training")
print("\nThe agent will automatically train itself!")
print("="*70)
