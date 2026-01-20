# AEGIS Usage Examples

## Example 1: Basic Autonomous Operation

```python
from aegis_autonomous import AutonomousAEGIS, AEGISConfig

# Create AEGIS with default configuration
config = AEGISConfig(
    vocab_size=1000,
    d_model=256,
    population_size=5
)

aegis = AutonomousAEGIS(config)

# The agent starts with built-in curiosity about:
# - Neural architectures
# - Reasoning strategies
# - Learning efficiency

# Start autonomous operation
# Agent will autonomously:
# 1. Generate goals based on curiosity
# 2. Ask questions
# 3. Search for knowledge
# 4. Propose improvements
aegis.start_autonomous_operation(
    max_iterations=50,
    think_interval_seconds=10
)
```

**Expected Output:**
```
=======================================================================
STARTING AUTONOMOUS OPERATION
=======================================================================

--- Iteration 1/50 ---
Agent decided: web_search
🔍 Performing web search: 'What is optimal attention mechanisms for reasoning?'
✓ Found and integrated 3 knowledge items
Agent state: 2 active goals, 3 knowledge items, 0 pending questions

--- Iteration 2/50 ---
Agent decided: ask_human
🤔 🤔 🤔 ...
AGENT HAS A QUESTION
Question: How does hierarchical vs flat reasoning approaches work?
Motivation: I want to understand reasoning_strategies better
...
```

## Example 2: Interactive Session with Agent

```python
from aegis_autonomous import AutonomousAEGIS, AEGISConfig

config = AEGISConfig()
aegis = AutonomousAEGIS(config)

# Start interactive session
aegis.interactive_session()
```

**Interactive Commands:**

```
>>> status
Operation Statistics:
  Questions asked: 5
  Web searches performed: 8
  Improvements proposed: 2
  Goals completed: 1
...

>>> goals
Agent Goals: {
  'current_goal': {
    'description': 'Learn about neural_architecture',
    'progress': 0.3,
    'type': 'knowledge_acquisition'
  },
  'active_goals': 2,
  'completed_goals': 1
}

>>> ask what are the benefits of hierarchical reasoning?
Agent: I'll think about that and search for information...
🔍 Performing web search: 'what are the benefits of hierarchical reasoning?'
Agent: Found and integrated 3 knowledge items

>>> tell Hierarchical reasoning allows for abstraction and decomposition of complex problems
Agent: Thank you! I've added that to my knowledge base (ID: k_42)

>>> knowledge
Knowledge Base: {
  'total_items': 15,
  'topics': 5,
  'verified_items': 3,
  'sources': {'web_search_general': 8, 'web_search_academic': 4, 'human': 3},
  'avg_confidence': 0.82
}
```

## Example 3: Supervised Evolution

```python
from aegis_system import AEGIS, AEGISConfig
import torch

config = AEGISConfig(
    population_size=10,
    elite_ratio=0.3,
    max_generations=20,
    require_approval_for_code_gen=True
)

aegis = AEGIS(config)

# Define custom evaluation function
def evaluate_on_reasoning_tasks(model):
    """Evaluate model on reasoning benchmarks"""

    # Example: Test on arithmetic reasoning
    # ... your evaluation code ...

    return {
        'accuracy': 0.85,
        'efficiency': 0.90,
        'generalization': 0.78
    }

# Run evolution
stats = aegis.evolve(
    num_generations=20,
    evaluation_function=evaluate_on_reasoning_tasks
)

# Review results
for gen_stat in stats:
    print(f"Generation {gen_stat['generation']}: "
          f"Best={gen_stat['best_score']:.3f}, "
          f"Mean={gen_stat['mean_score']:.3f}")
```

**With Approval:**

```python
# When system requests approval for new architecture:
pending = aegis.approval_manager.get_pending_requests()

for request in pending:
    print(f"\nRequest: {request.title}")
    print(f"Risk: {request.risk_assessment}")
    print(f"Changes: {request.proposed_changes}")

    # Review and approve
    aegis.approval_manager.approve_request(
        request_id=request.request_id,
        reviewer_name="Researcher",
        approval_code="approved_gen_5_variation_2",
        notes="Architecture looks reasonable, proceeding with evaluation"
    )
```

## Example 4: Knowledge Augmentation

```python
from aegis_autonomous import AutonomousAEGIS

aegis = AutonomousAEGIS()

# Agent autonomously searches for knowledge
query = "latest research on linear attention mechanisms"
results = aegis.knowledge_system.search_and_learn(
    query=query,
    topic="attention_mechanisms",
    search_type="academic"
)

print(f"Added {len(results)} knowledge items")

# Synthesize knowledge from multiple sources
synthesis_id = aegis.knowledge_system.synthesize_knowledge(
    topic="attention_mechanisms"
)

# Get knowledge summary
summary = aegis.knowledge_system.get_knowledge_summary("attention_mechanisms")
print(f"Topic: {summary['topic']}")
print(f"Total items: {summary['total_items']}")
print(f"Average confidence: {summary['avg_confidence']:.2f}")

# Export knowledge
export = aegis.knowledge_system.export_knowledge(topic="attention_mechanisms")
with open('attention_knowledge.json', 'w') as f:
    json.dump(export, f, indent=2)
```

## Example 5: Safety Monitoring and Emergence Detection

```python
from aegis_autonomous import AutonomousAEGIS
import torch

aegis = AutonomousAEGIS()

# Set up custom emergence alert handler
def custom_alert_handler(message, severity):
    """Custom handler for emergence alerts"""
    print(f"[{severity.upper()}] {message}")

    if severity == 'critical':
        # Send email, Slack notification, etc.
        send_alert_to_operator(message)

aegis.emergence_detector.alert_callback = custom_alert_handler

# Monitor during training
model = aegis.reasoning_engine

for step in range(1000):
    # Training step
    loss, grad_norm, runtime = train_step(model)

    # Monitor for anomalies
    safe_to_continue = aegis.emergence_detector.monitor_training_step(
        loss=loss,
        gradient_norm=grad_norm,
        runtime=runtime
    )

    if not safe_to_continue:
        print("Training frozen due to anomaly!")
        break

    # Periodically evaluate capabilities
    if step % 100 == 0:
        evaluation_suite = {
            'logical_reasoning': eval_logical_reasoning,
            'pattern_recognition': eval_patterns,
            'abstraction': eval_abstraction
        }

        emergent_capabilities = aegis.emergence_detector.evaluate_capabilities(
            model,
            evaluation_suite
        )

        if emergent_capabilities:
            print(f"Detected {len(emergent_capabilities)} emergent capabilities!")
            # System automatically freezes and notifies operator

# Check emergence detector status
status = aegis.emergence_detector.get_status_report()
print(f"System frozen: {status['is_frozen']}")
print(f"Emergent capabilities detected: {status['emergent_capabilities_detected']}")
```

## Example 6: Custom Goals and Curiosity

```python
from aegis_autonomous import AutonomousAEGIS
from core.agency.autonomous_agent import GoalType

aegis = AutonomousAEGIS()

# Register custom knowledge gaps to trigger curiosity
aegis.agent.curiosity.register_knowledge_gap(
    topic="quantum_computing",
    gap_description="quantum advantage in optimization problems"
)

aegis.agent.curiosity.register_knowledge_gap(
    topic="neuromorphic_computing",
    gap_description="spiking neural networks for energy efficiency"
)

# Generate custom goals
aegis.agent.goal_generator.generate_goal(
    goal_type=GoalType.UNDERSTANDING,
    trigger="Research interest",
    context={
        'concept': 'quantum-inspired optimization for neural architecture search',
        'priority': 0.9
    }
)

# Let agent pursue these goals
action = aegis.agent.think()
print(f"Agent decided: {action}")

result = aegis.agent.execute_action(action)
print(f"Result: {result}")
```

## Example 7: Multi-Generation Evolution with Checkpointing

```python
from aegis_system import AEGIS, AEGISConfig
import torch

config = AEGISConfig(max_generations=100)
aegis = AEGIS(config)

# Evolve in batches with checkpointing
for batch in range(10):
    print(f"\n=== Evolution Batch {batch + 1}/10 ===")

    # Evolve 10 generations
    stats = aegis.evolve(num_generations=10)

    # Save checkpoint
    checkpoint = {
        'generation': aegis.evolution_framework.current_generation,
        'population': aegis.evolution_framework.population,
        'stats': stats,
        'reasoning_engine_state': aegis.reasoning_engine.state_dict()
    }

    torch.save(checkpoint, f'aegis_checkpoint_batch_{batch}.pt')

    # Get best model so far
    best_candidate = max(
        aegis.evolution_framework.population,
        key=lambda c: c.performance_metrics.get('accuracy', 0)
    )

    print(f"Best so far: {best_candidate.description}")
    print(f"Performance: {best_candidate.performance_metrics}")

    # Check for plateau
    if len(stats) > 5:
        recent_scores = [s['best_score'] for s in stats[-5:]]
        improvement = max(recent_scores) - min(recent_scores)

        if improvement < 0.01:
            print("Performance plateau detected")
            # Could adjust mutation rate, etc.

# Final deployment
best_model = aegis.deploy_best_model()
print(f"Deployment request: {best_model.approval_request_id}")
```

## Example 8: Custom Evaluation Function

```python
from aegis_system import AEGIS
import torch
import torch.nn as nn

aegis = AEGIS()

# Define comprehensive evaluation
def comprehensive_evaluation(model: nn.Module) -> dict:
    """Evaluate model on multiple dimensions"""

    results = {}

    # 1. Task Performance
    # Test on various reasoning tasks
    results['arithmetic_reasoning'] = evaluate_arithmetic(model)
    results['logical_reasoning'] = evaluate_logic(model)
    results['pattern_completion'] = evaluate_patterns(model)

    # 2. Efficiency
    # Measure inference time and memory
    import time
    test_input = torch.randint(0, 1000, (16, 128))

    start = time.time()
    with torch.no_grad():
        _ = model(test_input)
    inference_time = time.time() - start

    results['inference_speed'] = 1.0 / (inference_time + 1e-6)

    # 3. Generalization
    # Test on held-out distribution
    results['ood_performance'] = evaluate_out_of_distribution(model)

    # 4. Robustness
    # Test with noisy inputs
    results['noise_robustness'] = evaluate_robustness(model)

    # 5. Composite Score
    results['accuracy'] = (
        0.4 * results['arithmetic_reasoning'] +
        0.3 * results['logical_reasoning'] +
        0.2 * results['pattern_completion'] +
        0.1 * results['ood_performance']
    )

    return results

# Use in evolution
stats = aegis.evolve(
    num_generations=30,
    evaluation_function=comprehensive_evaluation
)
```

## Example 9: Emergency Procedures

```python
from aegis_autonomous import AutonomousAEGIS

aegis = AutonomousAEGIS()

# Start autonomous operation
aegis.start_autonomous_operation(max_iterations=1000)

# ... Later, if something concerning happens ...

# Pause autonomous operation
aegis.pause_autonomous_operation("Need to review recent changes")

# Check what happened
status = aegis.get_system_status()
print(status['evolution'])
print(status['safety'])

# Review approvals
pending = aegis.approval_manager.get_pending_requests()
for req in pending:
    print(f"{req.title}: {req.description}")

# If everything looks good, resume
aegis.resume_autonomous_operation("approved_by_operator_123")

# Or, if something is seriously wrong
aegis.emergency_stop("Unexpected behavior detected")

# This freezes everything and requires manual intervention to resume
```

## Example 10: Integration with External Systems

```python
from aegis_autonomous import AutonomousAEGIS
import requests

class ExternalIntegration:
    """Integrate AEGIS with external systems"""

    def __init__(self, aegis):
        self.aegis = aegis

        # Set up callbacks for external notifications
        aegis.approval_manager.register_notification_callback(
            self.send_slack_notification
        )

        aegis.emergence_detector.alert_callback = self.send_email_alert

    def send_slack_notification(self, message, request):
        """Send approval request to Slack"""
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

        payload = {
            "text": f"🔔 AEGIS Approval Required",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message}
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Review"},
                            "url": f"http://your-aegis-dashboard/approvals/{request.request_id}"
                        }
                    ]
                }
            ]
        }

        requests.post(webhook_url, json=payload)

    def send_email_alert(self, message, severity):
        """Send email for critical alerts"""
        if severity == 'critical':
            # Use your email service
            send_email(
                to="operator@example.com",
                subject=f"[CRITICAL] AEGIS Alert",
                body=message
            )

# Use integration
aegis = AutonomousAEGIS()
integration = ExternalIntegration(aegis)

# Now all approvals and alerts go to Slack/Email
aegis.start_autonomous_operation()
```
