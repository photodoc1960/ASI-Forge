# HRM Training: Current Status and Missing Pieces

## **CRITICAL: HRM is NOT Currently Trained**

The HRM (Hierarchical Reasoning Model) is currently **untrained** (random weights). This is why we use GPT-2 for agent decisions.

## Current System Architecture

### What EXISTS:
1. ✅ **HRM Model Definition** (`core/hrm/hierarchical_reasoning.py`)
   - Dual-timescale processing (slow planning, fast execution)
   - Recurrent planning and execution modules
   - Adaptive computation time
   - ~1-5M parameters depending on config

2. ✅ **TrainingValidator** (`core/evolution/code_execution.py:389-492`)
   - Does 100 training steps on synthetic data
   - **Purpose**: Validates model CAN train (no NaN gradients, loss doesn't explode)
   - **NOT actual training** - just validation!
   - Uses Adam optimizer, CrossEntropyLoss
   - Checks for gradient issues

3. ✅ **Evolution Framework** (`core/evolution/supervised_evolution.py`)
   - Generates architecture variations
   - Validates safety
   - Validates trainability (100 steps)
   - **BUT: Never actually trains the models!**

### What's MISSING:

❌ **No Real Training Loop!**
- No dataset loading
- No multi-epoch training
- No checkpointing/saving
- No convergence monitoring
- No evaluation on validation set

## How HRM Training SHOULD Work

Based on the existing code structure, here's what needs to be added:

### Option 1: Supervised Learning on Agent Actions

**When**: After the agent (using GPT-2) makes decisions

**Process**:
```python
# 1. Agent makes decision using GPT-2
action = agent.think()  # Uses GPT-2

# 2. Encode agent state as tokens
agent_state_tokens = tokenizer.encode_agent_state(agent.state)

# 3. Target action token
target_action = tokenizer.encode_action(action)

# 4. Train HRM to predict this
optimizer.zero_grad()
hrm_output = hrm(agent_state_tokens)
loss = criterion(hrm_output, target_action)
loss.backward()
optimizer.step()
```

**Training Data**: Agent's own decision logs
- Input: Agent state (goals, knowledge, curiosity)
- Output: Action taken (web_search, ask_human, propose_improvement)

**File location**: Would need to add to `aegis_autonomous.py:start_autonomous_operation()`

### Option 2: Reinforcement Learning from Outcomes

**When**: Continuously during autonomous operation

**Process**:
```python
# 1. Agent proposes action using HRM
action = hrm.decide(state)

# 2. Execute action
result = execute(action)

# 3. Calculate reward
reward = calculate_reward(result)
# +1 if knowledge gained, goal achieved
# -1 if action failed, no progress
# +10 if improvement approved

# 4. Update HRM
optimizer.zero_grad()
policy_loss = -log_prob * reward  # REINFORCE algorithm
policy_loss.backward()
optimizer.step()
```

**Advantages**:
- Learns what actions lead to success
- No need for labeled data
- Directly optimizes for agent goals

**File location**: Would need to add to `core/agency/autonomous_agent.py`

### Option 3: Imitation Learning from GPT-2

**When**: Run as a separate training phase

**Process**:
```python
# Generate training dataset
training_data = []
for scenario in scenarios:
    # GPT-2 makes decision
    gpt2_action = gpt2_agent.decide(scenario)

    # Store as training example
    training_data.append((scenario, gpt2_action))

# Train HRM to imitate GPT-2
for epoch in range(num_epochs):
    for scenario, target_action in training_data:
        hrm_action = hrm(scenario)
        loss = cross_entropy(hrm_action, target_action)
        loss.backward()
        optimizer.step()
```

**Advantages**:
- Distills GPT-2 knowledge into smaller HRM
- HRM learns from expert (GPT-2)
- Once trained, can replace GPT-2

**File location**: New file `core/training/hrm_trainer.py`

## What About Evolution?

The evolution framework (`supervised_evolution.py`) is designed for **architecture search**, not training:

1. Generates code variations (add layer, modify attention)
2. Validates they compile and can train (100 steps)
3. Evaluates them (but on what? No real task!)
4. Selects best architectures

**Problem**: Even when evolution finds a "better architecture", it's still untrained!

## Recommended Implementation

Here's what I recommend adding:

### 1. Create HRM Trainer Class

```python
# core/training/hrm_trainer.py

class HRMTrainer:
    """Trains HRM using agent decision logs"""

    def __init__(self, hrm_model, optimizer, device='cpu'):
        self.model = hrm_model
        self.optimizer = optimizer
        self.device = device
        self.training_buffer = []

    def log_decision(self, state, action, reward):
        """Log agent decision for training"""
        self.training_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now()
        })

    def train_batch(self, batch_size=32):
        """Train on batch from buffer"""
        if len(self.training_buffer) < batch_size:
            return None

        # Sample batch
        batch = random.sample(self.training_buffer, batch_size)

        # Train
        total_loss = 0
        for item in batch:
            self.optimizer.zero_grad()

            # Forward pass
            predicted_action = self.model(item['state'])

            # Loss
            loss = criterion(predicted_action, item['action'])

            # Weighted by reward (good actions get more weight)
            loss = loss * (1.0 + item['reward'])

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / batch_size
```

### 2. Integrate into Autonomous Operation

```python
# In aegis_autonomous.py

def start_autonomous_operation(self, max_iterations=100):
    # ... existing code ...

    # Add HRM trainer
    if hasattr(self, 'reasoning_engine'):
        hrm_trainer = HRMTrainer(
            self.reasoning_engine,
            torch.optim.AdamW(self.reasoning_engine.parameters(), lr=1e-4)
        )

    for iteration in range(max_iterations):
        # Agent thinks (using GPT-2)
        action = self.agent.think()

        # Execute action
        result = self.agent.execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(action, result)

        # Log for HRM training
        hrm_trainer.log_decision(
            self.agent.get_state(),
            action,
            reward
        )

        # Train HRM periodically
        if iteration % 10 == 0:
            loss = hrm_trainer.train_batch()
            if loss:
                logger.info(f"HRM training loss: {loss:.4f}")
```

### 3. Evaluation and Switching

```python
def _evaluate_hrm_performance(self):
    """Check if HRM is ready to replace GPT-2"""

    # Test on validation scenarios
    hrm_correct = 0
    gpt2_correct = 0

    for scenario in validation_scenarios:
        hrm_action = self.reasoning_engine.decide(scenario)
        gpt2_action = self.agent.reasoning_engine.decide(scenario)

        # Check which matches optimal action
        if hrm_action == optimal_action:
            hrm_correct += 1
        if gpt2_action == optimal_action:
            gpt2_correct += 1

    hrm_accuracy = hrm_correct / len(validation_scenarios)
    gpt2_accuracy = gpt2_correct / len(validation_scenarios)

    # Switch if HRM is 90% as good as GPT-2
    if hrm_accuracy >= 0.9 * gpt2_accuracy:
        logger.info(f"HRM ready! Accuracy: {hrm_accuracy:.2%} vs GPT-2: {gpt2_accuracy:.2%}")
        return True

    return False
```

## Timeline

If we implement this:

1. **Iteration 1-100**: Agent uses GPT-2, HRM trains in background
2. **Iteration 100**: Evaluate HRM performance
3. **Iteration 100+**: If HRM good enough, switch agent to use HRM
4. **Evolution**: Now when agent proposes improvements, they improve the HRM it's actually using!

## Summary

**Current State**:
- ❌ HRM exists but is untrained (random weights)
- ✅ GPT-2 makes decisions (pretrained)
- ✅ Evolution generates variations but doesn't train them
- ✅ TrainingValidator checks trainability (100 steps) but doesn't do real training

**What's Needed**:
1. Create `HRMTrainer` class to train on agent decision logs
2. Integrate into autonomous operation loop
3. Add reward calculation based on action outcomes
4. Periodically evaluate HRM vs GPT-2 performance
5. Switch from GPT-2 to HRM when ready

**Then**: The autonomous agent will be using its own trained reasoning model that it evolved and improved itself!

**Files to Create/Modify**:
- `core/training/hrm_trainer.py` (NEW)
- `aegis_autonomous.py` (ADD training integration)
- `core/agency/autonomous_agent.py` (ADD reward calculation)

Would you like me to implement this training infrastructure?
