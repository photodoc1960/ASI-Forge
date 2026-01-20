"""
Quick test of pretrained LLM agent integration
"""

import logging
from aegis_autonomous import AutonomousAEGIS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("Testing AEGIS with Pretrained LLM")
print("=" * 70)

# Initialize with pretrained LLM
print("\nInitializing AEGIS with pretrained LLM...")
aegis = AutonomousAEGIS(use_pretrained_llm=True)

print("\n" + "=" * 70)
print("Testing agent.think() with pretrained LLM")
print("=" * 70)

# Let the agent think 3 times
for i in range(3):
    print(f"\n--- Think cycle {i+1} ---")
    action = aegis.agent.think()

    print(f"Action: {action['action']}")
    print(f"Confidence: {action.get('confidence', 'N/A')}")
    if 'reasoning' in action:
        print(f"Reasoning: {action['reasoning']}")
    if 'query' in action:
        print(f"Query: {action['query']}")
    if 'question' in action:
        print(f"Question: {action['question']}")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
