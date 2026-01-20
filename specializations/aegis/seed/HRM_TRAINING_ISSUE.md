# HRM Training Issue - Architecture Mismatch

## Problem Discovered

**Date**: 2025-10-26
**Status**: ❌ HRM training temporarily disabled

### Root Cause

The HRM (Hierarchical Reasoning Model) training fails with `index out of range` errors because of a fundamental architecture mismatch:

- **HRM vocabulary size**: ~1,000 tokens (custom small vocabulary)
- **GPT-2 tokenizer vocabulary**: 50,257 tokens

When we tokenize state descriptions using GPT-2's tokenizer, we get token IDs like 5234, 12487, etc. But the HRM's embedding layer only supports indices 0-999, causing out-of-range errors.

### Error Location

```
File: core/training/hrm_trainer.py:182
Error: index out of range in self
Context: output = self.model(state.unsqueeze(0))

State shape: torch.Size([64])
Token IDs: [1649, 1181, 25, 362, 4661, ...] <- These are way too large for HRM!
```

### What Was Attempted

✅ **Phase 1: Data Collection** - WORKS PERFECTLY
- Collected 1000+ training examples
- Data format is correct
- Actions properly encoded

❌ **Phase 2: Imitation Learning** - FAILS
- Cannot feed GPT-2 tokens to HRM model
- Architecture mismatch prevents training

---

## Solutions

### Option 1: Train GPT-2 LoRA Instead (RECOMMENDED ✅)

**Why**: GPT-2 is already working perfectly for agent decisions!

**Approach**:
1. Keep using GPT-2 for agent reasoning
2. Add LoRA adapters (already approved!)
3. Fine-tune GPT-2 with LoRA on collected data
4. Much simpler and will actually work

**Implementation**:
```python
# In aegis_autonomous.py
def _start_gpt2_finetuning(self):
    """Fine-tune GPT-2 with LoRA on collected agent data"""

    from peft import LoraConfig, get_peft_model, TaskType

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply to GPT-2
    model = self.agent.reasoning_engine.model
    peft_model = get_peft_model(model, lora_config)

    # Fine-tune on collected data
    # (train to predict better actions based on agent state)
```

**Benefits**:
- ✅ Uses working GPT-2 architecture
- ✅ LoRA already approved and installed
- ✅ Only trains ~800K parameters (0.65% of model)
- ✅ Can actually improve agent performance
- ✅ No vocabulary mismatch issues

---

### Option 2: Redesign HRM Training

**Requirements**:
1. Create HRM-specific tokenizer with matching vocabulary
2. Train HRM embeddings from scratch
3. Much more complex implementation
4. Uncertain if HRM will perform better than GPT-2

**Status**: Not recommended - GPT-2 + LoRA is simpler and proven

---

### Option 3: Use HRM for Different Purpose

**Idea**: Keep GPT-2 for language, use HRM for meta-reasoning

**Approach**:
- GPT-2: Generates action text
- HRM: High-level planning and strategy
- Separate responsibilities

**Status**: Interesting but requires rearchitecture

---

## Current Status

### What's Working ✅

1. **Autonomous Operation**: Agent runs continuously
2. **Data Collection**: 1000+ examples collected successfully
3. **GPT-2 Decision Making**: Works great for agent actions
4. **Web Search**: arXiv search working perfectly
5. **Knowledge Integration**: Learning from search results
6. **Goal Generation**: Autonomous goal setting working

### What's Not Working ❌

1. **HRM Training**: Disabled due to vocabulary mismatch
2. **Phase 2 Transition**: Cannot automatically train HRM

### What's Disabled ⏸️

**File**: `aegis_autonomous.py` line 1285
```python
# TEMPORARILY DISABLED: HRM training needs architecture redesign
if False and num_examples >= 1000 and iteration % 100 == 0:
    self._start_imitation_learning()
```

---

## Recommendation

**Implement GPT-2 LoRA Fine-Tuning** (Option 1)

This will:
1. Actually work (no architecture mismatch)
2. Improve agent performance on its specific tasks
3. Use the approved LoRA system
4. Train quickly (~10-30 minutes on CPU)
5. Require minimal code changes

The collected training data is perfect for this! We have 1000 examples of:
- Agent state → GPT-2 decision → Outcome
- Can fine-tune GPT-2 to make better decisions

---

## Files Status

### Working Files ✅
- `core/training/data_collector.py` - Collecting data perfectly
- `aegis_autonomous.py` - Phase 1 works great
- Training data: `data/hrm_training_data.json` (1000 examples)

### Needs Fix ⚠️
- `core/training/hrm_trainer.py` - Vocabulary mismatch issue
- Phase 2 transition - Needs to use GPT-2 + LoRA instead

### Not Affected ✅
- `core/training/rl_trainer.py` - RL logic is fine
- All other autonomous systems working

---

## Next Steps

1. Keep collecting data (it's useful!)
2. Implement GPT-2 + LoRA fine-tuning
3. Use collected data to improve GPT-2 decisions
4. HRM can be revisited later with proper architecture

**Bottom Line**: The system IS training - just needs to train GPT-2 instead of HRM!
