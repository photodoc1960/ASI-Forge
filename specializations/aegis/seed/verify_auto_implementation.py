#!/usr/bin/env python
"""
Verification script to prove automatic implementation works
This demonstrates that approved improvements execute AUTOMATICALLY
"""

import sys
import logging
from aegis_autonomous import AutonomousAEGIS
from core.auto_configure import AutoConfigurator
from interfaces.human_approval import ChangeType

# Set up logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("\n" + "="*70)
print("VERIFICATION: Automatic Implementation Chain")
print("="*70)
print("\nThis test proves that approved improvements execute AUTOMATICALLY")
print("with NO manual intervention required.\n")

# Initialize AEGIS
print("Step 1: Initializing AEGIS system...")
config = AutoConfigurator.load_config()
aegis = AutonomousAEGIS(config)
print(f"✓ System initialized")
print(f"  - Approval callbacks registered: {len(aegis.approval_manager.on_approval_callbacks)}")
print(f"  - Callback function: {aegis.approval_manager.on_approval_callbacks[0].__name__}")

# Create a test approval request
print("\nStep 2: Creating approval request...")
request_id = aegis.approval_manager.request_approval(
    change_type=ChangeType.ARCHITECTURE_MODIFICATION,
    title="Add Hierarchical Attention Layer to HRM [TEST]",
    description="Testing automatic implementation of layer addition",
    rationale="Verify that callback chain executes and implements change automatically",
    risk_assessment={
        'risk_level': 'low',
        'reversibility': True,
        'impact': 'minimal'
    },
    proposed_changes={
        'type': 'add_layer',
        'current_layers': config.high_level_layers,
        'new_layers': config.high_level_layers + 1
    },
    reversibility=True,
    estimated_impact="low"
)
print(f"✓ Request created: {request_id}")

# Get current architecture state
current_layers = config.high_level_layers
current_params = sum(p.numel() for p in aegis.reasoning_engine.parameters())
print(f"\n  Current architecture:")
print(f"    - Layers: {current_layers}")
print(f"    - Parameters: {current_params:,}")

# Approve the request - this should trigger AUTOMATIC implementation
print(f"\nStep 3: Approving request (should trigger AUTOMATIC implementation)...")
print("-" * 70)
success = aegis.approval_manager.approve_request(
    request_id=request_id,
    reviewer_name="Verification Test",
    approval_code="TEST-AUTO-IMPL-001",
    notes="Testing automatic implementation chain"
)
print("-" * 70)

if not success:
    print("\n❌ FAILED: Approval did not succeed")
    sys.exit(1)

print(f"\n✓ Approval processed")

# Verify implementation happened
print(f"\nStep 4: Verifying implementation happened AUTOMATICALLY...")
new_layers = aegis.config.high_level_layers
new_params = sum(p.numel() for p in aegis.reasoning_engine.parameters())

print(f"\n  Updated architecture:")
print(f"    - Layers: {new_layers}")
print(f"    - Parameters: {new_params:,}")

# Check if changes were applied
if new_layers > current_layers:
    print(f"\n✅ SUCCESS: Architecture was AUTOMATICALLY updated!")
    print(f"   Layers increased from {current_layers} → {new_layers}")
    print(f"   Parameters increased from {current_params:,} → {new_params:,}")
    print("\n" + "="*70)
    print("VERIFICATION PASSED: Automatic implementation is WORKING")
    print("="*70)
    print("\nThe system is FULLY AUTOMATED - no manual steps required!")
    sys.exit(0)
else:
    print(f"\n❌ FAILED: Architecture was NOT updated")
    print(f"   Layers: {current_layers} → {new_layers} (no change)")
    print("\n" + "="*70)
    print("VERIFICATION FAILED: Automatic implementation did NOT work")
    print("="*70)
    sys.exit(1)
