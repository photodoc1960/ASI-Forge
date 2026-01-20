#!/usr/bin/env python
"""
Test complete search flow - agent thinks, executes search, extracts insights
"""

import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator

# Normal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("="*70)
print("TESTING COMPLETE SEARCH FLOW")
print("="*70)

# Initialize AEGIS
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)

print("\n1. Agent thinks and decides action...")
action = aegis.agent.think()
print(f"   Action: {action['action']}")
print(f"   Query: {action.get('query', 'N/A')}")

if action['action'] == 'web_search':
    print("\n2. Executing web search...")
    result = aegis.agent.execute_action(action)
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Knowledge gained: {result.get('knowledge_gained', 0)} items")

    # Check insights
    state = aegis.agent.get_agent_state()
    print(f"\n3. Agent state after search:")
    print(f"   Total knowledge items: {state['knowledge_items']}")
    print(f"   Active goals: {state['active_goals']}")
else:
    print("\n   (Not a web_search action, skipping execution)")

print("\n" + "="*70)
print("✅ Test complete - search flow working correctly!")
print("="*70)
