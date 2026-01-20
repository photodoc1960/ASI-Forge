"""Check current AEGIS configuration"""

from aegis_autonomous import AutonomousAEGIS

aegis = AutonomousAEGIS(use_pretrained_llm=False)

print("Current Configuration:")
print(f"  High-level layers: {aegis.config.high_level_layers}")
print(f"  Low-level layers: {aegis.config.low_level_layers}")
print(f"  Total parameters: {sum(p.numel() for p in aegis.reasoning_engine.parameters()):,}")
print(f"  Model type: {type(aegis.reasoning_engine).__name__}")

print("\nApproval Manager:")
print(f"  Callbacks registered: {len(aegis.approval_manager.on_approval_callbacks)}")
print(f"  Pending requests: {len(aegis.approval_manager.get_pending_requests())}")
