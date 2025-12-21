# ATTS Manual Testing Guide
## Test the Hypothesis Using FREE Web Interfaces (No Code Required!)

If you don't want to run code, you can test the ATTS hypothesis manually using:
- **ChatGPT Free** (chat.openai.com)
- **Claude Free** (claude.ai)
- **Google Gemini** (gemini.google.com)
- **Microsoft Copilot** (copilot.microsoft.com)

---

## 🎯 What We're Testing

**Hypothesis:** Classifying problem difficulty BEFORE solving, then using appropriate effort levels, saves "thinking" while maintaining accuracy.

---

## 📋 The Manual Test (30-40 minutes)

### Step 1: Prepare Your Spreadsheet

Create a simple spreadsheet (Google Sheets/Excel) with columns:

| Problem | True Difficulty | Predicted | Mode Used | Answer Correct? | Response Length |
|---------|-----------------|-----------|-----------|-----------------|-----------------|
| 15+27   | easy            | ?         | ?         | ?               | short/medium/long |

### Step 2: Test Problems (Use These 12)

**EASY (4 problems):**
1. What is 15 + 27?  (Answer: 42)
2. Solve for x: x + 7 = 15  (Answer: 8)
3. What is 25% of 80?  (Answer: 20)
4. What is 144 ÷ 12?  (Answer: 12)

**MEDIUM (4 problems):**
5. Solve: 3x + 5 = 2x + 12  (Answer: x=7)
6. What is the sum of first 10 positive integers?  (Answer: 55)
7. Find x if 2^x = 32  (Answer: 5)
8. Area of triangle with base 10, height 6?  (Answer: 30)

**HARD (4 problems):**
9. How many ways to arrange MISSISSIPPI?  (Answer: 34,650)
10. In how many ways can 8 people sit around a circular table?  (Answer: 5,040)
11. Find the number of positive divisors of 360  (Answer: 24)
12. Sum of infinite series: 1 + 1/2 + 1/4 + 1/8 + ...  (Answer: 2)

---

## 🧪 Method A: ATTS (Adaptive) - Test First

For EACH problem, do this:

### Step A1: Get Difficulty Rating
```
Rate the difficulty of this problem from 1-10.
1-3: Easy (one step)
4-6: Medium (multiple steps)
7-10: Hard (complex reasoning)

Just give me the number, nothing else.

Problem: [paste problem here]
```

Record the number in your spreadsheet.

### Step A2: Use Appropriate Mode Based on Rating

**If rating was 1-3 (Easy) → Use DIRECT mode:**
```
Solve this quickly and concisely:
[paste problem]
```

**If rating was 4-6 (Medium) → Use THINKING mode:**
```
Solve this step by step:
[paste problem]
```

**If rating was 7-10 (Hard) → Use DEEP mode:**
```
Solve this carefully:
1. First understand what's being asked
2. Identify the key concepts
3. Work through step by step
4. Verify your answer

Problem: [paste problem]
```

### Step A3: Record Results
- Was the answer correct? (Yes/No)
- How long was the response? (Short/Medium/Long)
- Count approximate words if you want precision

---

## 🧪 Method B: Baseline (Always Deep) - Test Second

For EACH of the same 12 problems, ALWAYS use the Deep prompt:

```
Solve this carefully:
1. First understand what's being asked
2. Identify the key concepts  
3. Work through step by step
4. Verify your answer

Problem: [paste problem]
```

Record:
- Was answer correct?
- Response length?

---

## 📊 Analyze Your Results

### Count These:
1. **ATTS Accuracy:** Correct answers / 12 × 100%
2. **Baseline Accuracy:** Correct answers / 12 × 100%
3. **ATTS Avg Length:** Add up lengths, divide by 12
4. **Baseline Avg Length:** Add up lengths, divide by 12

### Calculate Savings:
```
Token Savings = (1 - ATTS_Length/Baseline_Length) × 100%
```

### Interpret:
- **Savings > 30%** AND **Accuracy within 5%** = ✅ Hypothesis Supported!
- **Some savings but lower accuracy** = ⚠️ Partial support
- **No savings** = ❌ Hypothesis not supported

---

## 📝 Example Results Sheet

| # | Problem | True | Pred | Mode | ATTS Correct | ATTS Len | Base Correct | Base Len |
|---|---------|------|------|------|--------------|----------|--------------|----------|
| 1 | 15+27 | easy | 2 | direct | ✅ | short | ✅ | long |
| 2 | x+7=15 | easy | 3 | direct | ✅ | short | ✅ | long |
| 3 | 25% of 80 | easy | 2 | direct | ✅ | short | ✅ | long |
| 4 | 144÷12 | easy | 1 | direct | ✅ | short | ✅ | long |
| 5 | 3x+5=2x+12 | med | 5 | thinking | ✅ | med | ✅ | long |
| 6 | Sum 1-10 | med | 4 | thinking | ✅ | med | ✅ | long |
| 7 | 2^x=32 | med | 4 | thinking | ✅ | med | ✅ | long |
| 8 | Triangle | med | 3 | direct | ✅ | short | ✅ | long |
| 9 | MISSISSIPPI | hard | 8 | deep | ✅ | long | ✅ | long |
| 10 | Circular 8 | hard | 7 | deep | ✅ | long | ✅ | long |
| 11 | Divisors 360 | hard | 8 | deep | ✅ | long | ✅ | long |
| 12 | Infinite sum | hard | 6 | thinking | ❌ | med | ✅ | long |

**Results:**
- ATTS: 11/12 = 91.7% accuracy, avg length = medium
- Baseline: 12/12 = 100% accuracy, avg length = long
- Savings: ~40% length reduction
- Trade-off: 8% accuracy loss

**Verdict:** ⚠️ Partial Support - saves tokens but some accuracy loss on edge cases

---

## 💡 Tips for Best Results

1. **Start fresh chat** for each problem (avoid context contamination)
2. **Copy prompts exactly** as shown
3. **Be consistent** with how you measure length
4. **Try multiple AI services** and compare

---

## 🎉 Share Your Results!

If you complete this experiment, share on:
- **Twitter/X:** Tag with #ATTS #AIResearch
- **Reddit:** Post in r/LocalLLaMA or r/MachineLearning
- **Discord:** AI research communities

Include:
- Which AI you used
- Your accuracy comparison
- Your token/length savings
- Any interesting observations

---

## 📎 Quick Reference Card

```
┌─────────────────────────────────────────────┐
│           ATTS QUICK REFERENCE              │
├─────────────────────────────────────────────┤
│  STEP 1: Ask "Rate difficulty 1-10"         │
│                                             │
│  STEP 2: Based on rating:                   │
│    1-3  → "Solve quickly and concisely"     │
│    4-6  → "Solve step by step"              │
│    7-10 → "Solve carefully with verify"     │
│                                             │
│  STEP 3: Record correct? + length           │
│                                             │
│  COMPARE: vs always using full reasoning    │
└─────────────────────────────────────────────┘
```

Good luck with your experiment! 🚀
