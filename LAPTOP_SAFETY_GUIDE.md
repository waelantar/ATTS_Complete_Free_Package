# Laptop Safety Guide - RTX 2050

## ⚠️ IMPORTANT: This is for RTX 2050 (4GB VRAM)

Your hardware is **modest but capable**. Follow these guidelines to avoid overheating, crashes, or data loss.

---

## 🛡️ Safety Features Built-In

The code now includes:

✅ **Auto-Checkpointing**: Progress saved every 10 problems
✅ **Safety Breaks**: 5-second pauses every 25 problems
✅ **Error Recovery**: Continues if one problem fails
✅ **Ctrl+C Protection**: Safe interrupt (keeps checkpoint)
✅ **Size Warnings**: Confirms before large datasets
✅ **Reduced k**: Pass@k uses k=2 (not k=3) by default
✅ **Refinement OFF**: Dialectical loops disabled by default

---

## 🚀 Quick Start (SAFE)

### Step 1: Convert MATH Dataset (Small Test)

```bash
# Quick test - 100 problems (~5-10 minutes)
python convert_math_dataset.py --size 100

# This creates: data/math_problems.json
```

### Step 2: Run Quick Test (5 problems only)

```bash
# Fastest - validate code works
python atts_experiment_local.py --quick-test

# Expected runtime: ~2 minutes
# Output: results/comprehensive_results_*.json
```

### Step 3: Run Small Experiment (Safe)

```bash
# Safe first run - 25 problems
python atts_experiment_local.py --max-problems 25

# Expected runtime: ~10-15 minutes
```

### Step 4: Larger Runs (Carefully)

```bash
# Medium run - 100 problems
python atts_experiment_local.py --max-problems 100

# Expected runtime: ~40-60 minutes
# Monitor temperature!
```

---

## 📊 Dataset Sizes & Runtime Estimates

| Size | Estimated Time | Safety | Recommendation |
|------|----------------|--------|----------------|
| 5 (--quick-test) | 2 min | ✅ Very Safe | First test |
| 25 (--max-problems 25) | 10-15 min | ✅ Safe | Validate full workflow |
| 100 (--max-problems 100) | 40-60 min | ⚠️ Monitor | Good for paper validation |
| 500 | 3-5 hours | ⚠️⚠️ Long | Break into batches |
| 1000+ | 6+ hours | ❌ Risky | Not recommended |

**Rule of Thumb**: ~0.5 minutes per problem (without refinement)

---

## 🌡️ Temperature Monitoring

### Check GPU Temperature

```bash
# Windows (if NVIDIA drivers installed)
nvidia-smi

# Look for "Temp" column - should stay under 85°C
```

### If Getting Hot (>80°C)

1. **Take a break**: Ctrl+C to stop (checkpoint auto-saves)
2. **Reduce dataset**: Use `--max-problems 50` instead of 100
3. **Increase breaks**: Edit `SAFETY_BREAK_INTERVAL` to 10 in code
4. **Disable refinement**: Already OFF by default (good!)

---

## 💾 Progress Checkpointing

### How It Works

- Auto-saves every 10 problems to `results/checkpoint_atts_*.json`
- If crash/interrupt, you can resume from checkpoint (manual analysis)
- On successful completion, checkpoint auto-deleted

### If Interrupted

```bash
# Your progress is saved! Check:
ls results/checkpoint_atts_*.json

# You can analyze partial results or re-run:
python atts_experiment_local.py --quick-test
```

---

## ⚡ Performance Optimization Flags

### Faster Runs (Trade-offs)

```bash
# Disable escalation (saves ~10% time, slight accuracy drop)
python atts_experiment_local.py --no-escalation --max-problems 100

# Reduce Pass@k samples (saves ~15% time)
python atts_experiment_local.py --passk-k 1 --max-problems 100

# Combine both (saves ~25% time)
python atts_experiment_local.py --no-escalation --passk-k 1 --max-problems 100
```

### Slower But More Thorough (Enable Refinement)

```bash
# WARNING: 2-3x slower! Only for final validation
python atts_experiment_local.py --enable-refinement --max-problems 25

# Expected: ~30-45 minutes for 25 problems
```

---

## 🔧 Troubleshooting

### "Ollama Error: Connection refused"

```bash
# Check if Docker container is running
docker ps

# If not running:
docker start ollama

# If container doesn't exist:
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull qwen2.5:3b-instruct
```

### "Out of Memory" / Crash

- **Solution 1**: Reduce dataset size (`--max-problems 25`)
- **Solution 2**: Reduce Pass@k (`--passk-k 1`)
- **Solution 3**: Restart Docker (`docker restart ollama`)

### Laptop Getting Very Hot

- **Immediate**: Ctrl+C to stop (progress auto-saved)
- **Next run**: Use smaller batches (`--max-problems 50`)
- **Check cooling**: Ensure laptop vents not blocked
- **Consider**: External cooling pad

### Progress Lost

- **Check**: `results/checkpoint_atts_*.json`
- **Prevention**: Checkpoints every 10 problems (automatic)
- **Manual save**: Results saved at end to `comprehensive_results_*.json`

---

## 📋 Recommended Workflow

### Day 1: Validation

```bash
# 1. Quick test (2 min)
python atts_experiment_local.py --quick-test

# 2. Small run (15 min)
python atts_experiment_local.py --max-problems 25

# Check results look good
```

### Day 2: Full Validation

```bash
# Morning: Medium run (1 hour)
python atts_experiment_local.py --max-problems 100

# Afternoon: Ablation studies
python atts_experiment_local.py --no-escalation --max-problems 50
python atts_experiment_local.py --passk-k 1 --max-problems 50
```

### Optional: Enable Refinement (Slow)

```bash
# Only if you need full dialectical loops
python atts_experiment_local.py --enable-refinement --max-problems 25

# Expected: 30-45 minutes for 25 problems
```

---

## 🎯 Hardware-Specific Settings

### RTX 2050 (Your GPU)

- **VRAM**: 4GB (enough for qwen2.5:3b)
- **Safe k**: 2 (default)
- **Safe batch**: 100 problems
- **Refinement**: OFF by default

### If You Had More VRAM (8GB+)

```bash
# Could use k=3 safely
python atts_experiment_local.py --passk-k 3 --max-problems 100

# Could enable refinement more often
python atts_experiment_local.py --enable-refinement --max-problems 50
```

### If You Had Less VRAM (2GB)

```bash
# Would need k=1
python atts_experiment_local.py --passk-k 1 --max-problems 50

# Smaller model
python atts_experiment_local.py --model qwen2.5:1.5b-instruct
```

---

## ✅ Safety Checklist Before Long Runs

- [ ] Docker container running (`docker ps`)
- [ ] Model downloaded (`docker exec -it ollama ollama list`)
- [ ] Dataset converted (`ls data/math_problems.json`)
- [ ] Quick test passed (`--quick-test`)
- [ ] Laptop on stable surface (good airflow)
- [ ] Charger plugged in
- [ ] Temperature monitoring ready (`nvidia-smi`)
- [ ] Realistic time estimate (0.5 min/problem)

---

## 📞 If Things Go Wrong

1. **Ctrl+C** to stop safely
2. Check `results/checkpoint_atts_*.json` for partial results
3. Reduce dataset size for next run
4. Check temperature
5. Restart Ollama if needed: `docker restart ollama`

---

## 🎓 Understanding the Outputs

### What's Being Measured

- **Token Savings**: How much cheaper ATTS is vs always-Deep
- **Accuracy Cost**: How much worse ATTS performs vs always-Deep
- **Pareto Improvement**: Saves >20% tokens with <5% accuracy drop

### Good Results Look Like

```
💰 Token Savings: 40-50%
📉 Accuracy Cost: 2-4%
✅ Pareto Improvement: YES
```

### Warning Signs

```
💰 Token Savings: <20%  ← Not efficient enough
📉 Accuracy Cost: >10%  ← Too much accuracy lost
❌ Pareto Improvement: NO
```

---

**Last Updated**: Dec 21, 2025
**Target Hardware**: RTX 2050 (4GB VRAM)
**Model Tested**: qwen2.5:3b-instruct
