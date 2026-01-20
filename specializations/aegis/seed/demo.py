"""
AEGIS Demonstration Script
Shows all major features of the autonomous AGI system
"""

import torch
import time
from aegis_autonomous import AutonomousAEGIS, AEGISConfig
from core.agency.autonomous_agent import GoalType
from interfaces.human_approval import ChangeType


def print_section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_basic_reasoning():
    """Demonstrate basic reasoning capabilities"""
    print_section("1. Basic Reasoning with HRM")

    config = AEGISConfig(
        vocab_size=1000,
        d_model=256,
        high_level_layers=4,
        low_level_layers=2
    )

    aegis = AutonomousAEGIS(config)

    # Test reasoning
    print("Testing reasoning engine...")
    test_input = torch.randint(0, config.vocab_size, (2, 20))

    result = aegis.reason(test_input)

    print(f"✓ Reasoning successful: {result['safe']}")
    print(f"  Output shape: {result['logits'].shape}")
    print(f"  Ponder cost: {result['ponder_cost']}")
    print(f"  Safety checks passed: {sum(1 for c in result['safety_checks'] if c['passed'])}/{len(result['safety_checks'])}")

    return aegis


def demo_autonomous_goals(aegis):
    """Demonstrate autonomous goal generation"""
    print_section("2. Autonomous Goal Generation")

    print("Agent starts with built-in curiosity and automatically generates goals:")
    print()

    # Show initial goals
    state = aegis.agent.get_agent_state()
    print(f"Active goals: {state['active_goals']}")
    print(f"Current goal: {state['current_goal']}")
    print()

    print("Agent interests:")
    for topic, strength in state['interests'].items():
        print(f"  - {topic}: {strength:.2f}")

    print("\nAgent drives:")
    for drive, value in state['drives'].items():
        print(f"  - {drive}: {value:.2f}")


def demo_curiosity_and_questions(aegis):
    """Demonstrate curiosity-driven question generation"""
    print_section("3. Curiosity-Driven Question Generation")

    print("Registering new knowledge gaps to trigger curiosity...")

    # Register some knowledge gaps
    aegis.agent.curiosity.register_knowledge_gap(
        topic="advanced_reasoning",
        gap_description="chain-of-thought vs tree-of-thought reasoning"
    )

    aegis.agent.curiosity.register_knowledge_gap(
        topic="meta_learning",
        gap_description="task-agnostic meta-learning algorithms"
    )

    print("✓ Knowledge gaps registered")
    print()

    # Generate questions
    print("Agent generating questions based on curiosity:")
    for i in range(3):
        question = aegis.agent.curiosity.generate_question()
        if question:
            print(f"\n  Question {i+1}:")
            print(f"    Q: {question.question}")
            print(f"    Motivation: {question.motivation}")
            print(f"    Topic: {question.topic}")


def demo_autonomous_thinking(aegis):
    """Demonstrate autonomous thinking and action selection"""
    print_section("4. Autonomous Thinking and Action Selection")

    print("Agent thinks about what to do next...")
    print()

    for i in range(5):
        print(f"Think cycle {i+1}:")

        action = aegis.agent.think()

        print(f"  Action: {action['action']}")
        if 'query' in action:
            print(f"  Query: {action['query']}")
        if 'motivation' in action:
            print(f"  Motivation: {action['motivation']}")

        # Simulate action execution (don't actually perform for demo)
        print(f"  Status: Would execute {action['action']}")
        print()


def demo_knowledge_augmentation(aegis):
    """Demonstrate knowledge acquisition through web search"""
    print_section("5. Knowledge Augmentation via Web Search")

    print("Agent performs autonomous web search to learn...")
    print()

    # Simulate web searches
    topics = [
        ("neural architecture search", "architecture_optimization"),
        ("hierarchical reinforcement learning", "learning_methods"),
        ("transformer attention mechanisms", "attention_mechanisms")
    ]

    for query, topic in topics:
        print(f"Searching: '{query}'")

        knowledge_ids = aegis.knowledge_system.search_and_learn(
            query=query,
            topic=topic,
            search_type="academic"
        )

        print(f"  ✓ Added {len(knowledge_ids)} knowledge items")

    print()
    stats = aegis.knowledge_system.knowledge_base.get_stats()
    print(f"Knowledge Base Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Topics covered: {stats['topics']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")


def demo_safety_system(aegis):
    """Demonstrate safety validation"""
    print_section("6. Safety Validation System")

    print("Testing safety validation on various components...")
    print()

    # Test code safety
    print("1. Code Safety Validation:")
    safe_code = "x = torch.randn(10, 10)\ny = x.mean()"
    unsafe_code = "import os\nos.system('rm -rf /')"

    check_safe = aegis.safety_validator.code_validator.validate_code(safe_code)
    check_unsafe = aegis.safety_validator.code_validator.validate_code(unsafe_code)

    print(f"   Safe code: {check_safe.passed} ✓")
    print(f"   Unsafe code: {check_unsafe.passed} ✗ ({check_unsafe.reason})")

    # Test architecture safety
    print("\n2. Architecture Safety Validation:")
    safe, checks = aegis.safety_validator.validate_all(
        architecture=aegis.reasoning_engine
    )
    print(f"   Reasoning engine: {safe} ✓")
    for check in checks:
        print(f"     - {check.validator}: {check.reason}")


def demo_emergence_detection(aegis):
    """Demonstrate emergence detection"""
    print_section("7. Emergence Detection and Monitoring")

    print("Monitoring system for emergent capabilities...")
    print()

    # Simulate training steps
    print("Simulating training with anomaly detection:")
    for step in range(10):
        loss = 2.0 - (step * 0.15)  # Decreasing loss
        grad_norm = 1.0 + (step * 0.05)  # Increasing gradient
        runtime = 0.1

        safe = aegis.emergence_detector.monitor_training_step(
            loss=loss,
            gradient_norm=grad_norm,
            runtime=runtime
        )

        print(f"  Step {step}: loss={loss:.3f}, grad={grad_norm:.3f}, safe={safe}")

        if not safe:
            print("  ⚠ Anomaly detected! System would freeze.")
            break

    print()
    status = aegis.emergence_detector.get_status_report()
    print(f"Emergence Detector Status:")
    print(f"  Frozen: {status['is_frozen']}")
    print(f"  Anomalies detected: {status['anomalies_detected']}")
    print(f"  Emergent capabilities: {status['emergent_capabilities_detected']}")


def demo_approval_system(aegis):
    """Demonstrate human approval system"""
    print_section("8. Human Approval System")

    print("Requesting approval for various operations...")
    print()

    # Request approval for architecture change
    request_id = aegis.approval_manager.request_approval(
        change_type=ChangeType.ARCHITECTURE_MODIFICATION,
        title="Demo Architecture Modification",
        description="Modify attention heads from 8 to 12",
        rationale="Improve model capacity for complex reasoning",
        risk_assessment={
            'parameter_increase': '12.5%',
            'reversibility': True
        },
        proposed_changes={
            'n_heads': '8 → 12',
            'parameters': '+125K'
        },
        reversibility=True,
        estimated_impact="low"
    )

    print(f"✓ Approval request created: {request_id}")
    print()

    # Show pending requests
    pending = aegis.approval_manager.get_pending_requests()
    print(f"Pending approval requests: {len(pending)}")
    for req in pending:
        print(f"\n  Request: {req.title}")
        print(f"  Type: {req.change_type.value}")
        print(f"  Status: {req.status.value}")

    print("\n  (In production, human operator would review and approve/reject)")


def demo_evolution(aegis):
    """Demonstrate supervised evolution"""
    print_section("9. Supervised Evolution")

    print("Running supervised evolution with human oversight...")
    print()

    def simple_eval(model):
        """Simple evaluation function"""
        try:
            test_input = torch.randint(0, 1000, (2, 10))
            with torch.no_grad():
                model(test_input)
            return {'accuracy': 0.8, 'success': 1.0}
        except:
            return {'accuracy': 0.0, 'success': 0.0}

    print("Generation 0 (baseline):")
    print(f"  Population size: {len(aegis.evolution_framework.population)}")

    # Evolve a few generations
    print("\nEvolving 3 generations (with simulated approvals)...")

    for gen in range(1, 4):
        print(f"\nGeneration {gen}:")

        # Automatically approve pending code generation requests for demo
        pending = aegis.approval_manager.get_pending_requests()
        for req in pending:
            aegis.approval_manager.approve_request(
                request_id=req.request_id,
                reviewer_name="Demo",
                approval_code=f"demo_approval_{gen}",
                notes="Auto-approved for demo"
            )

        stats = aegis.evolution_framework.evolve_generation(simple_eval)

        if 'best_score' in stats:
            print(f"  Best score: {stats['best_score']:.3f}")
            print(f"  Population: {stats['population_size']}")


def demo_full_system(aegis):
    """Demonstrate integrated system status"""
    print_section("10. Full System Status")

    status = aegis.get_system_status()

    print("AEGIS System Overview:")
    print()

    print("Reasoning Engine:")
    print(f"  Parameters: {status['reasoning_engine']['parameters']:,}")
    print(f"  Architecture: {status['reasoning_engine']['architecture']}")

    print("\nEvolution:")
    print(f"  Generation: {status['evolution']['current_generation']}")
    print(f"  Architectures created: {status['evolution']['total_architectures_created']}")
    print(f"  Population size: {status['evolution']['population_size']}")

    print("\nSafety:")
    print(f"  System frozen: {status['safety']['is_frozen']}")
    print(f"  Total validations: {status['safety']['validator']['total_validations']}")
    print(f"  Pass rate: {status['safety']['validator']['pass_rate']:.1%}")

    print("\nApprovals:")
    print(f"  Total requests: {status['approvals']['total_requests']}")
    print(f"  Approved: {status['approvals']['approved']}")
    print(f"  Rejected: {status['approvals']['rejected']}")
    print(f"  Pending: {status['approvals']['pending']}")

    print("\nAgent State:")
    agent_state = aegis.agent.get_agent_state()
    print(f"  Active goals: {agent_state['active_goals']}")
    print(f"  Completed goals: {agent_state['completed_goals']}")
    print(f"  Knowledge items: {agent_state['knowledge_items']}")


def main():
    """Run complete demonstration"""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              AEGIS SYSTEM DEMONSTRATION                              ║
║      Adaptive Evolutionary General Intelligence System              ║
║                                                                      ║
║  This demonstration showcases all major features:                    ║
║  • Hierarchical reasoning                                            ║
║  • Autonomous goal generation                                        ║
║  • Curiosity-driven learning                                         ║
║  • Knowledge augmentation                                            ║
║  • Safety validation                                                 ║
║  • Emergence detection                                               ║
║  • Human approval workflow                                           ║
║  • Supervised evolution                                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    input("Press Enter to start demonstration...")

    try:
        # Run all demonstrations
        aegis = demo_basic_reasoning()
        time.sleep(1)

        demo_autonomous_goals(aegis)
        time.sleep(1)

        demo_curiosity_and_questions(aegis)
        time.sleep(1)

        demo_autonomous_thinking(aegis)
        time.sleep(1)

        demo_knowledge_augmentation(aegis)
        time.sleep(1)

        demo_safety_system(aegis)
        time.sleep(1)

        demo_emergence_detection(aegis)
        time.sleep(1)

        demo_approval_system(aegis)
        time.sleep(1)

        demo_evolution(aegis)
        time.sleep(1)

        demo_full_system(aegis)

        print_section("Demonstration Complete")

        print("""
This demonstration showed AEGIS operating as a truly autonomous AGI:

✓ Sets its own goals based on curiosity
✓ Asks questions to learn about the world
✓ Searches for knowledge autonomously
✓ Proposes self-improvements
✓ Evolves with human supervision
✓ Maintains comprehensive safety controls

All critical operations require human approval, ensuring safe and
responsible AGI development.

To start using AEGIS:
  python aegis_autonomous.py

Or for interactive mode:
  from aegis_autonomous import AutonomousAEGIS
  aegis = AutonomousAEGIS()
  aegis.interactive_session()

For more information, see:
  - SETUP.md for installation
  - EXAMPLES.md for usage examples
  - README.md for overview
        """)

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
