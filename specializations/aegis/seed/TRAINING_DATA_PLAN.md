# HRM Training Data: Practical Implementation Plan

## Current Reality
**NO training data exists!** The system has:
- ✅ HRM model definition
- ✅ Agent that makes decisions (using GPT-2)
- ❌ NO labeled training examples
- ❌ NO datasets for agent tasks

## Recommended Approach: 3-Phase Training

### Phase 1: Data Collection (Bootstrap from GPT-2)

**Duration**: 1000-5000 autonomous iterations (~2-10 hours)

**What happens**:
```python
# In aegis_autonomous.py - add data collection mode

class AutonomousAEGIS:
    def __init__(self, config, collect_training_data=True):
        # ... existing code ...

        if collect_training_data:
            self.training_data_collector = TrainingDataCollector()

    def start_autonomous_operation(self, max_iterations=1000):
        for iteration in range(max_iterations):
            # Capture state BEFORE decision
            state_vector = self._encode_agent_state()

            # GPT-2 makes decision
            action = self.agent.think()

            # Execute action
            result = self.agent.execute_action(action)

            # Log as training example
            self.training_data_collector.add_example(
                state=state_vector,
                action=action,
                result=result
            )

        # Save to disk
        self.training_data_collector.save('hrm_training_data.json')
```

**State Encoding** (what to capture):
```python
def _encode_agent_state(self):
    return {
        # Goals
        'active_goals': [g.description for g in self.agent.goal_generator.active_goals[:5]],
        'goal_types': [g.goal_type.value for g in self.agent.goal_generator.active_goals[:5]],

        # Curiosity
        'curiosity_count': len(self.agent.curiosity.curiosity_queue),
        'top_curiosity': [c.gap_description for c in list(self.agent.curiosity.curiosity_queue)[:3]],

        # Knowledge
        'knowledge_count': len(self.knowledge_system.knowledge_base.knowledge),
        'recent_knowledge': [k.content[:100] for k in recent_knowledge[:3]],

        # History
        'recent_actions': [a['action'] for a in self.agent.action_history[-5:]],
        'questions_asked': self.operation_stats['questions_asked'],
        'searches_performed': self.operation_stats['searches_performed'],

        # Context
        'iteration': iteration,
        'timestamp': datetime.now().isoformat()
    }
```

**Output**: `hrm_training_data.json` with 1000-5000 examples

**Example**:
```json
{
  "examples": [
    {
      "state": {
        "active_goals": ["Deeply understand my own architecture"],
        "curiosity_count": 3,
        "top_curiosity": ["efficient attention mechanisms"],
        "knowledge_count": 5,
        "recent_actions": ["web_search", "idle", "ask_human"]
      },
      "action": {
        "action": "web_search",
        "query": "transformer attention mechanisms",
        "confidence": 0.67
      },
      "result": {
        "status": "completed",
        "knowledge_gained": 5
      }
    }
  ]
}
```

### Phase 2: Supervised Imitation Learning

**Duration**: 10-50 epochs (~1-2 hours)

**Train HRM to imitate GPT-2**:
```python
# core/training/hrm_trainer.py (NEW FILE)

import torch
import torch.nn as nn
import json

class HRMTrainer:
    def __init__(self, hrm_model, device='cuda'):
        self.model = hrm_model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def load_training_data(self, filepath):
        """Load data collected from GPT-2"""
        with open(filepath) as f:
            data = json.load(f)
        return data['examples']

    def encode_state(self, state_dict):
        """Convert state dict to tensor"""
        # Tokenize text descriptions
        goal_tokens = self.tokenize_goals(state_dict['active_goals'])
        curiosity_tokens = self.tokenize_curiosity(state_dict['top_curiosity'])

        # Numeric features
        numeric = torch.tensor([
            state_dict['curiosity_count'],
            state_dict['knowledge_count'],
            state_dict['questions_asked'],
            state_dict['searches_performed']
        ])

        # Action history (one-hot)
        action_history = self.encode_action_history(state_dict['recent_actions'])

        # Concatenate all features
        state_tensor = torch.cat([
            goal_tokens,
            curiosity_tokens,
            numeric,
            action_history
        ])

        return state_tensor

    def encode_action(self, action_dict):
        """Convert action to class index"""
        action_map = {
            'web_search': 0,
            'ask_human': 1,
            'propose_improvement': 2,
            'idle': 3
        }
        return action_map[action_dict['action']]

    def train_epoch(self, examples):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0

        for example in examples:
            # Encode state
            state = self.encode_state(example['state']).to(self.device)

            # Encode target action
            target = torch.tensor([self.encode_action(example['action'])]).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(state.unsqueeze(0))  # Add batch dimension

            # Get action logits
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output

            # Loss
            loss = self.criterion(logits, target)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predicted = logits.argmax(dim=-1)
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(examples)
        accuracy = correct / len(examples)

        return avg_loss, accuracy

    def train(self, data_file, epochs=20):
        """Full training loop"""
        examples = self.load_training_data(data_file)

        print(f"Training on {len(examples)} examples for {epochs} epochs")

        for epoch in range(epochs):
            loss, acc = self.train_epoch(examples)
            print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, Accuracy={acc:.2%}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'hrm_epoch_{epoch+1}.pt')

    def save_checkpoint(self, filepath):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, filepath)
```

**Expected Result**: HRM reaches 60-80% accuracy in imitating GPT-2

### Phase 3: Reinforcement Fine-tuning

**Duration**: Ongoing during autonomous operation

**Switch to HRM and improve with RL**:
```python
# In aegis_autonomous.py

def start_autonomous_operation_with_rl(self, max_iterations=5000):
    """Use trained HRM and continue improving with RL"""

    # Switch agent to use HRM instead of GPT-2
    self.agent.reasoning_engine = self.reasoning_engine  # HRM
    self.agent.use_pretrained_llm = False

    for iteration in range(max_iterations):
        # HRM makes decision
        state = self._encode_agent_state()
        action = self.agent.think()  # Now using HRM!

        # Execute
        result = self.agent.execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(action, result)

        # Update HRM with policy gradient
        self._rl_update(state, action, reward)
```

**Reward Function**:
```python
def _calculate_reward(self, action, result):
    """Calculate reward for action outcome"""

    reward = 0.0

    # Web search rewards
    if action['action'] == 'web_search':
        if result.get('knowledge_gained', 0) > 0:
            reward += 1.0 * result['knowledge_gained']
        else:
            reward -= 0.5  # Wasted search

    # Improvement proposal rewards
    elif action['action'] == 'propose_improvement':
        if result.get('status') == 'approved':
            reward += 10.0  # Big reward!
        elif result.get('status') == 'already_pending':
            reward -= 2.0  # Duplicate proposal
        elif result.get('status') == 'pending_approval':
            reward += 1.0  # At least created valid proposal

    # Ask human rewards
    elif action['action'] == 'ask_human':
        if result.get('status') == 'completed':
            reward += 0.5
        else:
            reward -= 0.1

    # Idle penalty
    elif action['action'] == 'idle':
        reward -= 0.1

    return reward
```

## Timeline

### Week 1: Data Collection
- Run autonomous operation for 5000 iterations
- Collect `hrm_training_data.json` (5000 examples)
- GPT-2 remains the decision maker

### Week 2: Imitation Learning
- Create `core/training/hrm_trainer.py`
- Train HRM for 20 epochs
- Evaluate: HRM should reach 70%+ accuracy
- Save checkpoint: `hrm_trained.pt`

### Week 3+: RL Fine-tuning
- Switch agent to use trained HRM
- Continue autonomous operation with RL updates
- HRM improves beyond GPT-2 through experience

## File Structure

```
/mnt/d/aegis/
├── data/
│   ├── hrm_training_data.json      (5000 examples from GPT-2)
│   └── validation_scenarios.json   (100 hand-crafted test cases)
├── checkpoints/
│   ├── hrm_epoch_5.pt
│   ├── hrm_epoch_10.pt
│   └── hrm_final.pt
├── core/
│   └── training/
│       ├── __init__.py
│       ├── hrm_trainer.py          (Imitation learning)
│       ├── rl_trainer.py           (RL fine-tuning)
│       └── data_collector.py       (Collects from autonomous operation)
```

## Validation

Test HRM performance against GPT-2:
```python
def evaluate_hrm_vs_gpt2():
    """Compare HRM and GPT-2 on validation scenarios"""

    scenarios = load_validation_scenarios()

    hrm_correct = 0
    gpt2_correct = 0

    for scenario in scenarios:
        # HRM decision
        hrm_action = hrm.decide(scenario)

        # GPT-2 decision
        gpt2_action = gpt2.decide(scenario)

        # Compare to optimal action (labeled by human)
        if hrm_action == scenario['optimal_action']:
            hrm_correct += 1
        if gpt2_action == scenario['optimal_action']:
            gpt2_correct += 1

    print(f"HRM accuracy: {hrm_correct/len(scenarios):.2%}")
    print(f"GPT-2 accuracy: {gpt2_correct/len(scenarios):.2%}")
```

## Summary

**Training Data Source**: Generated from GPT-2's decisions during autonomous operation

**No external datasets needed!** The system is self-contained:
1. GPT-2 acts as expert/teacher
2. HRM learns by imitation
3. HRM surpasses teacher through RL

**Ready to implement?** I can create:
1. `data_collector.py` - Logs GPT-2 decisions
2. `hrm_trainer.py` - Trains HRM on collected data
3. Integration into `aegis_autonomous.py` - Automatic data collection and training
