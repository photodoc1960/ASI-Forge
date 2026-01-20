# AEGIS User Guide: How to Interact with the Autonomous Agent

This guide explains how AEGIS actually works and how to interact with it.

---

## Understanding AEGIS Behavior

### Why the Agent is "Idle" Most of the Time

**This is normal and by design!** AEGIS is conservative to prevent uncontrolled behavior. The agent:

1. ✅ **Generates goals** based on curiosity (you saw this working)
2. ✅ **Proposes improvements** every 10 iterations (you saw this working)
3. ⚠️ **Decides on actions** but needs more specific triggers to execute actively

The goals like "Deeply understand my own architecture" are **abstract meta-goals**. The agent needs either:
- Human interaction to give it direction
- Approved improvement requests to execute on
- More specific knowledge gaps to explore

---

## How to Use AEGIS: 3 Modes

### Mode 1: Autonomous Operation (Observer Mode)

```bash
./run_aegis.sh
# Choose option 3: Autonomous operation
# Enter number of iterations: 50
```

**What happens**:
- Agent runs autonomously
- Proposes improvements every 10 iterations
- You see what it WANTS to do
- Nothing executed until you approve

**What you do**:
- Watch the logs
- Note the approval request IDs
- Approve them later in interactive mode

**Use this for**: Seeing what agent wants to do without interference

---

### Mode 2: Interactive Session (Recommended!)

```bash
python aegis_autonomous.py
# Choose option 2: Interactive session
```

**Now you can actually interact!**

#### Key Commands:

**See what the agent wants to do:**
```
>>> goals
```
Shows detailed goals with motivations, progress, and types.

**See pending approval requests:**
```
>>> pending
```
Shows all requests waiting for your approval with full details.

**Make the agent think and show you the decision:**
```
>>> think
```
Agent will decide an action and ask if you want to execute it.

**Approve a self-improvement request:**
```
>>> approve 4f7fe7e1-e6f3-40a2-b174-0cc3e9af0841
```
(Use the actual request ID from `pending` command)

**Ask the agent a question:**
```
>>> ask What is transformer architecture?
```
Agent will search the web and learn about it.

**Tell the agent something:**
```
>>> tell Neural architecture search is a method for automating model design
```
Agent adds this to its knowledge base.

**Check knowledge base:**
```
>>> knowledge
```

**Check system status:**
```
>>> status
```

---

### Mode 3: Direct Python Control (Advanced)

```python
from aegis_autonomous import AutonomousAEGIS

# Create agent
aegis = AutonomousAEGIS()

# Check what it wants to do
action = aegis.agent.think()
print(f"Agent wants to: {action}")

# Execute the action
if action['action'] != 'idle':
    result = aegis.agent.execute_action(action)
    print(f"Result: {result}")

# Check pending approvals
pending = aegis.approval_manager.get_pending_requests()
for req in pending:
    print(f"Request: {req.title}")

    # Approve it
    aegis.approval_manager.approve_request(
        request_id=req.request_id,
        reviewer_name="You",
        approval_code="manual_approval",
        notes="Approved for testing"
    )

# Run evolution after approval
aegis.evolution_framework.evolve_generation(
    evaluation_function=lambda model: {'accuracy': 0.8}
)
```

---

## Understanding the Agent's Decision-Making

### How Goals Work

When you see goals, they have types:

```
→ Goal 1: Deeply understand my own architecture and capabilities
   Type: understanding
   Progress: 0%
   Priority: 0.50
   Motivation: Intrinsic curiosity
```

**Goal Types:**
- `knowledge_acquisition` → Agent will ask questions or search
- `self_improvement` → Agent proposes architecture changes
- `exploration` → Agent searches for new information
- `understanding` → Agent tries to build mental models
- `capability_improvement` → Agent tries to get better at tasks

### How Actions Work

For each goal type, the agent decides actions:

**KNOWLEDGE_ACQUISITION goals** →
```python
{
    'action': 'web_search',  # or 'ask_human'
    'query': 'What is X?',
    'goal_id': 'goal_1',
    'motivation': 'I want to understand better'
}
```

**SELF_IMPROVEMENT goals** →
```python
{
    'action': 'propose_improvement',
    'aspect': 'architecture',
    'goal_id': 'goal_2',
    'motivation': 'Continuous improvement drive'
}
```

**The key**: Actions aren't executed unless:
1. The agent has specific knowledge gaps to fill, OR
2. You manually trigger with `think` command, OR
3. You ask the agent questions, OR
4. You approve improvement proposals

---

## Making the Agent More Active

Want to see more action? Here's how:

### 1. Use Interactive Mode and Guide It

```bash
python aegis_autonomous.py
# Option 2: Interactive session

>>> ask How does attention mechanism work in transformers?
🔍 Performing web search: 'How does attention mechanism work in transformers?'
✓ Found and integrated 2 knowledge items

>>> ask What is neural architecture search?
🔍 Performing web search: 'What is neural architecture search?'
✓ Found and integrated 2 knowledge items

>>> knowledge
Knowledge Base: {'total_items': 4, 'topics': 2, ...}

>>> goals
# Now the agent has MORE context and may generate more specific goals
```

### 2. Approve Improvement Proposals

```bash
>>> pending
PENDING APPROVAL REQUESTS (5)

  Request 1:
     ID: 4f7fe7e1-e6f3-40a2-b174-0cc3e9af0841
     Type: architecture_modification
     Title: Agent-Proposed Improvement: architecture
     ...

>>> approve 4f7fe7e1-e6f3-40a2-b174-0cc3e9af0841
✓ Request approved

# Now if you run evolution, it will actually modify the architecture
```

### 3. Use the `think` Command

```bash
>>> think
Agent is thinking...
  Decision: web_search
  query: What are optimal attention mechanisms for reasoning?
  goal_id: goal_1
  motivation: I want to understand neural_architecture better

Execute this action? (y/n): y
✓ Action executed: completed
```

---

## Example Interactive Session

```bash
$ python aegis_autonomous.py
# Choose 2: Interactive session

>>> help
# Read the commands

>>> goals
ACTIVE GOALS (2)

  → Goal 1: Deeply understand my own architecture and capabilities
     Type: understanding
     Progress: 0%
     Priority: 0.50
     Motivation: Intrinsic curiosity

  → Goal 2: Evolve architecture to better reasoning efficiency
     Type: self_improvement
     Progress: 0%
     Priority: 0.50
     Motivation: Intrinsic curiosity

>>> pending
PENDING APPROVAL REQUESTS (0)
  No pending requests

>>> think
Agent is thinking...
  Decision: idle
  reason: No action determined

# Agent is idle because goals are too abstract. Let's give it direction:

>>> ask What is hierarchical reasoning?
🔍 Performing web search: 'What is hierarchical reasoning?'
✓ Found and integrated 2 knowledge items

>>> ask What are latest advances in neural architecture search?
🔍 Performing web search: ...
✓ Found and integrated 2 knowledge items

>>> knowledge
Knowledge Base: {'total_items': 4, 'topics': 2, 'verified_items': 0, ...}

# Now the agent has learned something!

>>> tell You should focus on improving your attention mechanisms
Agent: Thank you! I've added that to my knowledge base (ID: k_5)

# After ~10 iterations in autonomous mode, agent will propose improvements

>>> pending
# Check for improvement proposals

>>> approve <request-id>
# Approve them

>>> status
# See overall system state
```

---

## Why Approvals are Required

**By design**, all self-modifications require human approval:

1. **Code generation** → Approval required
2. **Architecture changes** → Approval required
3. **Deployment** → Approval required

This prevents:
- ❌ Uncontrolled self-modification
- ❌ Runaway optimization
- ❌ Unexpected capabilities
- ❌ Resource exhaustion

You **must** approve changes for the agent to evolve. This is the core safety feature!

---

## Troubleshooting

### "Agent keeps saying 'idle'"

**Solutions**:
1. Use interactive mode and ask it questions
2. Tell it specific information to work with
3. Wait for improvement proposals (every 10 iterations)
4. Use the `think` command to see what it's considering

### "No pending approvals"

**Explanation**: Agent only proposes improvements every 10 iterations in autonomous mode. Either:
- Run autonomous mode for 20+ iterations
- Wait patiently
- Or: The agent is being very conservative (this is good!)

### "How do I know what to approve?"

**Check details**:
```bash
>>> pending
# Shows:
# - What the change is
# - Why the agent wants it
# - Risk assessment
# - What will be modified
```

Approve if:
- ✅ Change seems reasonable
- ✅ Risk is low/medium
- ✅ You understand what it does

Reject if:
- ❌ Unclear what it does
- ❌ Risk is high
- ❌ Agent is being too aggressive

### "I want the agent to be more active"

The current implementation is **conservative by design**. To make it more active, you would need to:

1. Give it more specific initial goals (edit `_bootstrap_agent_interests()`)
2. Lower the threshold for action execution
3. Add more knowledge gaps to explore
4. Implement a "curiosity scheduler" that forces exploration

**Or**: Use interactive mode and guide it with questions!

---

## Summary

**What AEGIS does autonomously:**
- ✅ Generates goals based on curiosity
- ✅ Proposes self-improvements (every 10 iterations)
- ✅ Makes decisions about what to do
- ✅ Maintains safety bounds

**What requires your input:**
- ⚙️ Approval for self-modifications
- ⚙️ Specific questions to explore
- ⚙️ Knowledge to integrate
- ⚙️ Direction for abstract goals

**Best way to use AEGIS:**
1. Start in **interactive mode** (option 2)
2. Use `goals` to see what it wants
3. Use `think` to see its decisions
4. Use `ask` to give it things to learn
5. Use `pending` to see approval requests
6. Use `approve` to let it evolve

**Remember**: The "idle" behavior is a **safety feature**, not a bug. It prevents uncontrolled autonomous action. You have full control!

---

For more details:
- `EXAMPLES.md` - Usage examples
- `API_SETUP.md` - API configuration
- `README.md` - System overview
