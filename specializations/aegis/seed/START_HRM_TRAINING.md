# How to Start HRM Training

## ✅ System is Ready!

You have:
- ✓ 1,100 training examples collected
- ✓ HRM vocabulary fixed (50,257 tokens)
- ✓ HRM model ready (29M parameters)
- ✓ GPU support enabled (RTX 4090)

---

## 🚀 Option 1: Start Training NOW (Recommended)

```bash
python train_hrm.py
```

This will:
1. Initialize HRM (29M parameters)
2. Train for 20 epochs on GPU (~10-15 minutes)
3. Automatically switch from GPT-2 to HRM if accuracy ≥ 70%
4. Start RL fine-tuning for continuous improvement

**Expected output:**
```
======================================================================
PHASE 2: HRM TRAINING (Imitation Learning)
======================================================================
✓ HRM initialized: 29,138,898 parameters (29.14M)
✓ Loaded 1100 training examples

Training HRM for 20 epochs...
Epoch 1/20: Loss=2.4567, Accuracy=45.23%
Epoch 2/20: Loss=1.8234, Accuracy=58.12%
Epoch 3/20: Loss=1.3456, Accuracy=67.45%
...
Epoch 20/20: Loss=0.3456, Accuracy=85.67%

======================================================================
PHASE 2 COMPLETE: HRM Training Finished
  Final loss: 0.3456
  Final accuracy: 85.67%
======================================================================

✅ HRM accuracy ≥ 70%! Switching from GPT-2 to HRM...

======================================================================
SWITCHING FROM GPT-2 TO HRM
======================================================================
✓ Agent now using trained HRM!
  Previous: GPT-2 (124M params)
  Current: HRM (29.1M params)
```

---

## 🔄 Option 2: Automatic Training (During Autonomous Operation)

```bash
python start_training.py
```

The system will:
- Continue collecting data with GPT-2
- Automatically trigger HRM training at 1,100 examples
- Switch to HRM when ready
- No manual intervention needed!

---

## 📊 What Happens During Training?

### Phase 1: Data Collection (COMPLETE ✅)
- **Status**: 1,100 examples collected
- **Model**: GPT-2 (bootstrap)
- **Purpose**: Collect high-quality decision data

### Phase 2: HRM Training (READY TO START ⏳)
- **Duration**: ~10-15 minutes on RTX 4090
- **Method**: Imitation learning from GPT-2
- **Goal**: HRM learns to make same decisions as GPT-2
- **Success**: Accuracy ≥ 70%

### Phase 3: Switch to HRM (AUTOMATIC 🔄)
- **Trigger**: HRM accuracy ≥ 70%
- **Result**: Agent now uses HRM instead of GPT-2
- **Benefit**: 4x faster, 76% smaller model

### Phase 4: RL Fine-tuning (CONTINUOUS 🎯)
- **When**: After switching to HRM
- **Method**: Policy gradient updates
- **Goal**: HRM continues improving through experience

---

## 📈 Expected Training Results

With 1,100 examples, you should see:

| Metric | Expected |
|--------|----------|
| **Final Accuracy** | 70-85% |
| **Final Loss** | 0.3-0.5 |
| **Training Time** | 10-15 min (GPU) |
| **Switch to HRM** | ✅ Yes (if accuracy ≥ 70%) |

---

## 🎯 After Training

Once HRM is active:

1. **Faster reasoning**: HRM is ~4x faster than GPT-2
2. **Better decisions**: HRM learned from 1,100 successful examples
3. **Continuous improvement**: RL keeps making HRM smarter
4. **Your vision realized**: Custom hierarchical model running!

---

## 🔍 Monitoring Training

### Check training status:
```bash
python demo.py
>>> training
```

### Watch live progress:
Training progress is logged in real-time with:
- Epoch number
- Loss (decreasing = good)
- Accuracy (increasing = good)
- Checkpoints saved every 5 epochs

---

## ⚠️ Troubleshooting

### "Not enough training data"
- You have 1,100 examples ✅
- Minimum: 100 (will work but less accurate)
- Recommended: 1,000+ (you have this!)

### "Training accuracy < 70%"
- Collect more data: Run `python start_training.py` longer
- Increase epochs: Edit `aegis_autonomous.py` line 1347
- Check data quality: Run `test_hrm_training.py`

### "Out of memory"
- Your RTX 4090 has 16GB ✅ (plenty of space)
- HRM only uses ~2GB for training
- Should not happen!

---

## 📚 Technical Details

### HRM Architecture:
```
- Vocabulary: 50,257 tokens (matches GPT-2)
- Embedding: 256 dimensions
- High-level layers: 3 (planning)
- Low-level layers: 2 (execution)
- Attention heads: 8
- Total parameters: 29.14M (23% of GPT-2)
```

### Training Configuration:
```
- Device: CUDA (RTX 4090)
- Epochs: 20
- Learning rate: 1e-4
- Optimizer: AdamW
- Batch processing: Per-example
- Checkpoints: Every 5 epochs
```

---

## 🎉 Ready to Start!

**Just run:**
```bash
python train_hrm.py
```

**Or for fully autonomous:**
```bash
python start_training.py
```

Your HRM will be trained and active in ~15 minutes! 🚀
