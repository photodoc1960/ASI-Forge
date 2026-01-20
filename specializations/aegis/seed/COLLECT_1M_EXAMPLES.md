# Collecting 1 Million Examples for HRM Training

## 🎯 Target: 1,000,000 Examples

With 29M parameter HRM, you need ~1M examples for proper training.

**Current**: 1,100 examples
**Needed**: 998,900 more examples
**Ratio goal**: 29 params/example ✅ Perfect!

---

## ⏱️ Time Required

### Single Instance:
- **Conservative** (5s/example): 1,387 hours (58 days)
- **Optimized** (2s/example): 555 hours (23 days)

### Parallel Collection (Recommended):
- **2 instances**: 277 hours (11.6 days)
- **4 instances**: 139 hours (5.8 days) ⭐ **Recommended**
- **8 instances**: 69 hours (2.9 days) ⭐⭐ **Fastest**

---

## 🚀 How to Collect Data Fast

### Option 1: Single Instance (Simple)

```bash
python collect_data_fast.py
```

Leave it running overnight/for days. Data auto-saves every 100 examples.

### Option 2: Parallel Instances (4x-8x Faster!)

**Terminal 1:**
```bash
python collect_data_fast.py
```

**Terminal 2:**
```bash
python collect_data_fast.py
```

**Terminal 3:**
```bash
python collect_data_fast.py
```

**Terminal 4:**
```bash
python collect_data_fast.py
```

All instances write to the same file - they'll merge automatically!

### Option 3: Background Collection (Linux/WSL)

```bash
# Start 4 background collectors
for i in {1..4}; do
  nohup python collect_data_fast.py > collector_$i.log 2>&1 &
done

# Check progress
watch -n 60 'wc -l data/hrm_training_data.json'

# Stop all collectors
pkill -f collect_data_fast.py
```

---

## 📊 Progress Monitoring

### Check Current Count:
```bash
python -c "
import json
with open('data/hrm_training_data.json') as f:
    data = json.load(f)
print(f'Examples: {len(data[\"examples\"]):,}')
print(f'Progress: {100*len(data[\"examples\"])/1000000:.2f}%')
"
```

### Or in Python:
```bash
python demo.py
>>> training
```

---

## 🎯 Collection Strategies

### Strategy 1: Continuous Collection (Recommended)
Run 4-8 parallel instances for 3-7 days straight.

**Pros:**
- Fastest path to 1M examples
- Most diverse data (many different scenarios)

**Cons:**
- Need to keep computer running

### Strategy 2: Incremental Collection
Run overnight, pause during day, resume at night.

**Pros:**
- Can use computer during day
- More flexible

**Cons:**
- Takes longer (2-3 weeks)

### Strategy 3: Distributed Collection
Run on multiple machines (if you have access).

**Pros:**
- Even faster!
- Less load per machine

**Cons:**
- Need to merge data files manually

---

## 🔧 Optimization Tips

### 1. Disable Unnecessary Logging
Edit `aegis_autonomous.py` - reduce logging verbosity to speed up:
```python
logging.basicConfig(level=logging.WARNING)  # Instead of INFO
```

### 2. Increase Iteration Speed
Agent makes ~1 decision per iteration. The faster iterations run, the more data collected.

### 3. Keep GPU Free
GPT-2 runs on GPU for faster reasoning = faster data collection

### 4. Monitor System Resources
```bash
# Check if system is bottlenecked
htop  # CPU usage
nvidia-smi -l 1  # GPU usage
```

---

## 📈 What Happens at 1M Examples?

Once you hit 1,000,000 examples:

1. ✅ **Automatic trigger**: System detects 1M examples
2. 🎓 **HRM training starts**: 20 epochs (~30-60 min on RTX 4090)
3. 🔄 **Auto-switch**: If accuracy ≥ 70%, switches from GPT-2 to HRM
4. 🚀 **Faster reasoning**: HRM is 4x faster than GPT-2
5. 📚 **Continuous learning**: RL keeps improving HRM

---

## 💾 Storage Requirements

### Data File Size:
- **1,100 examples**: ~500 KB
- **100,000 examples**: ~45 MB
- **1,000,000 examples**: ~450 MB

✅ Plenty of space on any modern system!

---

## ⚠️ Important Notes

### Data Auto-Saves
- Saves every 100 examples
- If script crashes, you don't lose progress
- Can safely Ctrl+C and resume later

### Parallel Safety
- Multiple instances can write to same file
- Data collector handles concurrent access
- No risk of corruption

### Progress Persistence
- Data saved to `data/hrm_training_data.json`
- Delete file to start fresh (NOT recommended!)
- Backup file periodically if paranoid

---

## 🎯 Quick Start Commands

### Start Collecting Now:
```bash
# Single instance
python collect_data_fast.py

# Or 4 parallel instances (in separate terminals)
python collect_data_fast.py  # Terminal 1
python collect_data_fast.py  # Terminal 2
python collect_data_fast.py  # Terminal 3
python collect_data_fast.py  # Terminal 4
```

### Check Progress:
```bash
python -c "
import json
with open('data/hrm_training_data.json') as f:
    data = json.load(f)
count = len(data['examples'])
print(f'{count:,} / 1,000,000 ({100*count/1000000:.2f}%)')
"
```

---

## 🎉 The Goal

**1 Million Examples** = **Properly Trained HRM**

With 1M examples and 29M parameters:
- ✅ No overfitting
- ✅ Good generalization
- ✅ Actually useful reasoning
- ✅ Your vision: custom AGI brain!

**Start collecting now - your HRM will thank you!** 🧠🚀
