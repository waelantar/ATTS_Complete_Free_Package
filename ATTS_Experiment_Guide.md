# ATTS Simulation Experiment: Complete Guide
## Adaptive Test-Time Scaling Validation Without Training Models

---

# TABLE OF CONTENTS

1. Overview & Goals
2. Prerequisites & Setup
3. Dataset Preparation
4. The Experiment Code
5. Running the Experiment Step-by-Step
6. Analyzing Results
7. Troubleshooting
8. Expected Results & Interpretation

---

# 1. OVERVIEW & GOALS

## What We're Testing
The core hypothesis of Adaptive Test-Time Scaling (ATTS) is:
> "Classifying problem difficulty BEFORE solving and routing to appropriate compute tiers 
>  saves significant tokens while maintaining high accuracy."

## What We'll Measure
- **Accuracy**: % of problems solved correctly
- **Token Usage**: Average output tokens per problem
- **Efficiency Ratio**: Accuracy per 1000 tokens
- **Classification Accuracy**: How well the model predicts difficulty

## Comparison
- **Baseline**: Force "Deep Thinking" mode on ALL problems
- **ATTS**: Route problems based on predicted difficulty

---

# 2. PREREQUISITES & SETUP

## Step 2.1: Install Python (if needed)
```bash
# Check if Python is installed
python3 --version

# If not installed (Ubuntu/Debian):
sudo apt update && sudo apt install python3 python3-pip
```

## Step 2.2: Create Project Directory
```bash
mkdir atts_experiment
cd atts_experiment
```

## Step 2.3: Install Required Packages
```bash
pip install openai anthropic tiktoken pandas matplotlib seaborn tqdm
```

## Step 2.4: Set Up API Keys
Create a file called `.env`:
```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

Or export directly:
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
```

---

# 3. DATASET PREPARATION

## Step 3.1: Create the Dataset File
Save this as `math_problems.json`:

```json
{
  "problems": [
    {
      "id": 1,
      "difficulty_label": "easy",
      "problem": "What is 15 + 27?",
      "answer": "42"
    },
    {
      "id": 2,
      "difficulty_label": "easy",
      "problem": "If x = 5, what is 2x + 3?",
      "answer": "13"
    },
    {
      "id": 3,
      "difficulty_label": "easy",
      "problem": "What is 144 divided by 12?",
      "answer": "12"
    },
    {
      "id": 4,
      "difficulty_label": "easy",
      "problem": "Solve for x: x + 7 = 15",
      "answer": "8"
    },
    {
      "id": 5,
      "difficulty_label": "easy",
      "problem": "What is 25% of 80?",
      "answer": "20"
    },
    {
      "id": 6,
      "difficulty_label": "medium",
      "problem": "Solve for x: 3x + 5 = 2x + 12",
      "answer": "7"
    },
    {
      "id": 7,
      "difficulty_label": "medium",
      "problem": "What is the area of a triangle with base 10 and height 6?",
      "answer": "30"
    },
    {
      "id": 8,
      "difficulty_label": "medium",
      "problem": "If f(x) = x^2 + 3x, what is f(4)?",
      "answer": "28"
    },
    {
      "id": 9,
      "difficulty_label": "medium",
      "problem": "Find the value of x if 2^x = 32",
      "answer": "5"
    },
    {
      "id": 10,
      "difficulty_label": "medium",
      "problem": "What is the sum of the first 10 positive integers?",
      "answer": "55"
    },
    {
      "id": 11,
      "difficulty_label": "medium",
      "problem": "Simplify: (x^2 - 9) / (x - 3)",
      "answer": "x + 3"
    },
    {
      "id": 12,
      "difficulty_label": "medium",
      "problem": "What is the probability of rolling a sum of 7 with two dice?",
      "answer": "1/6"
    },
    {
      "id": 13,
      "difficulty_label": "hard",
      "problem": "Find all integer solutions to x^2 + y^2 = 25 where x, y > 0",
      "answer": "(3,4) and (4,3)"
    },
    {
      "id": 14,
      "difficulty_label": "hard",
      "problem": "Prove that the sum of the angles in any triangle is 180 degrees, and find the third angle if two angles are 45° and 75°.",
      "answer": "60"
    },
    {
      "id": 15,
      "difficulty_label": "hard",
      "problem": "Find the derivative of f(x) = x^3 * ln(x) and evaluate at x = e",
      "answer": "4e^2"
    },
    {
      "id": 16,
      "difficulty_label": "hard",
      "problem": "How many ways can you arrange the letters in MISSISSIPPI?",
      "answer": "34650"
    },
    {
      "id": 17,
      "difficulty_label": "hard",
      "problem": "Find the sum of the infinite series: 1 + 1/2 + 1/4 + 1/8 + ...",
      "answer": "2"
    },
    {
      "id": 18,
      "difficulty_label": "hard",
      "problem": "In how many ways can 8 people be seated around a circular table?",
      "answer": "5040"
    },
    {
      "id": 19,
      "difficulty_label": "hard",
      "problem": "Find the integral of 1/(x^2 + 1) dx from 0 to 1",
      "answer": "π/4"
    },
    {
      "id": 20,
      "difficulty_label": "hard",
      "problem": "Find the number of positive divisors of 360",
      "answer": "24"
    }
  ]
}
```

## Step 3.2: Expand the Dataset (Optional)
For more robust results, expand to 100+ problems. Sources:
- GSM8K dataset (grade school math)
- MATH dataset (competition math)
- AoPS (Art of Problem Solving) archives

---

# 4. THE EXPERIMENT CODE

## Step 4.1: Create the Main Experiment Script
Save this as `atts_experiment.py`:

```python
"""
ATTS Simulation Experiment
Tests whether difficulty-based routing saves tokens while maintaining accuracy.

Usage:
    python atts_experiment.py --provider openai --model gpt-4o
    python atts_experiment.py --provider anthropic --model claude-3-5-sonnet-20241022
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tiktoken
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Difficulty thresholds (adjust these based on results)
THRESHOLD_DIRECT = 4      # d < 4 -> Direct mode
THRESHOLD_THINKING = 7    # 4 <= d < 7 -> Thinking mode
                          # d >= 7 -> Deep mode

# Escalation threshold
ESCALATION_THRESHOLD = 0.6  # If confidence < 0.6, escalate

# ============================================================================
# PROMPTS
# ============================================================================

DIFFICULTY_PROMPT = """Rate the difficulty of this math problem on a scale of 1-10.
- 1-3: Very easy (single step, basic arithmetic)
- 4-6: Medium (multiple steps, requires some thought)
- 7-10: Hard (complex reasoning, multiple concepts, competition-level)

IMPORTANT: Only output a single number from 1-10. Nothing else.

Problem: {problem}

Difficulty (1-10):"""

DIRECT_PROMPT = """Solve this math problem. Be concise.

Problem: {problem}

Answer:"""

THINKING_PROMPT = """Solve this math problem step by step.

Problem: {problem}

Let me work through this step by step:"""

DEEP_PROMPT = """Solve this math problem using careful, thorough reasoning.

Problem: {problem}

I'll solve this systematically:

1. First, let me understand what's being asked.
2. Then I'll identify the key concepts needed.
3. I'll work through the solution step by step.
4. Finally, I'll verify my answer by checking it.

Let me begin:"""

VERIFICATION_PROMPT = """I just solved a math problem. Rate my confidence in the answer from 0-10.
Consider: Did I make any calculation errors? Did I address all parts of the question?

Problem: {problem}
My solution: {solution}

Confidence (0-10):"""

# ============================================================================
# API CLIENTS
# ============================================================================

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_openai(client, model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call OpenAI API and return (response, token_count)"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    text = response.choices[0].message.content
    tokens = response.usage.completion_tokens
    return text, tokens

def call_anthropic(client, model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call Anthropic API and return (response, token_count)"""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    tokens = response.usage.output_tokens
    return text, tokens

def call_model(client, provider: str, model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Unified API caller"""
    if provider == "openai":
        return call_openai(client, model, prompt, max_tokens)
    elif provider == "anthropic":
        return call_anthropic(client, model, prompt, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ============================================================================
# CORE ATTS LOGIC
# ============================================================================

def estimate_difficulty(client, provider: str, model: str, problem: str) -> int:
    """Step 1: Estimate problem difficulty (1-10)"""
    prompt = DIFFICULTY_PROMPT.format(problem=problem)
    response, _ = call_model(client, provider, model, prompt, max_tokens=10)
    
    # Parse the number from response
    try:
        # Extract first number found
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            difficulty = int(numbers[0])
            return max(1, min(10, difficulty))  # Clamp to 1-10
    except:
        pass
    return 5  # Default to medium if parsing fails

def select_mode(difficulty: int) -> str:
    """Step 2: Select reasoning mode based on difficulty"""
    if difficulty < THRESHOLD_DIRECT:
        return "direct"
    elif difficulty < THRESHOLD_THINKING:
        return "thinking"
    else:
        return "deep"

def solve_problem(client, provider: str, model: str, problem: str, mode: str) -> Tuple[str, int]:
    """Step 3: Solve using selected mode"""
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 200
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 800
    else:  # deep
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 2000
    
    return call_model(client, provider, model, prompt, max_tokens)

def verify_solution(client, provider: str, model: str, problem: str, solution: str) -> float:
    """Step 4: Self-verification (returns confidence 0-1)"""
    prompt = VERIFICATION_PROMPT.format(problem=problem, solution=solution)
    response, _ = call_model(client, provider, model, prompt, max_tokens=10)
    
    try:
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            confidence = int(numbers[0]) / 10.0
            return max(0, min(1, confidence))
    except:
        pass
    return 0.7  # Default confidence

def check_answer(response: str, correct_answer: str) -> bool:
    """Check if response contains the correct answer"""
    response_lower = response.lower().replace(" ", "")
    answer_lower = correct_answer.lower().replace(" ", "")
    
    # Direct match
    if answer_lower in response_lower:
        return True
    
    # Try numeric comparison
    try:
        import re
        # Extract numbers from response
        response_nums = re.findall(r'-?\d+\.?\d*', response)
        answer_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
        
        if answer_nums:
            target = float(answer_nums[0])
            for num in response_nums:
                if abs(float(num) - target) < 0.01:
                    return True
    except:
        pass
    
    return False

# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_atts_experiment(client, provider: str, model: str, problems: List[Dict]) -> List[Dict]:
    """Run ATTS (adaptive) experiment"""
    results = []
    
    for prob in tqdm(problems, desc="ATTS Experiment"):
        # Step 1: Estimate difficulty
        difficulty = estimate_difficulty(client, provider, model, prob["problem"])
        
        # Step 2: Select mode
        mode = select_mode(difficulty)
        original_mode = mode
        
        # Step 3: Solve
        solution, tokens = solve_problem(client, provider, model, prob["problem"], mode)
        total_tokens = tokens
        
        # Step 4: Verify (for thinking/deep modes)
        escalated = False
        if mode in ["thinking", "deep"]:
            confidence = verify_solution(client, provider, model, prob["problem"], solution)
            
            # Escalation check
            if confidence < ESCALATION_THRESHOLD and mode != "deep":
                escalated = True
                mode = "deep"
                solution, extra_tokens = solve_problem(client, provider, model, prob["problem"], mode)
                total_tokens += extra_tokens
        
        # Check correctness
        correct = check_answer(solution, prob["answer"])
        
        results.append({
            "id": prob["id"],
            "problem": prob["problem"],
            "true_difficulty": prob["difficulty_label"],
            "predicted_difficulty": difficulty,
            "mode": mode,
            "original_mode": original_mode,
            "escalated": escalated,
            "tokens": total_tokens,
            "correct": correct,
            "solution": solution[:500]  # Truncate for storage
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    return results

def run_baseline_experiment(client, provider: str, model: str, problems: List[Dict]) -> List[Dict]:
    """Run baseline (always deep) experiment"""
    results = []
    
    for prob in tqdm(problems, desc="Baseline Experiment"):
        # Always use deep mode
        solution, tokens = solve_problem(client, provider, model, prob["problem"], "deep")
        correct = check_answer(solution, prob["answer"])
        
        results.append({
            "id": prob["id"],
            "problem": prob["problem"],
            "true_difficulty": prob["difficulty_label"],
            "mode": "deep",
            "tokens": tokens,
            "correct": correct,
            "solution": solution[:500]
        })
        
        time.sleep(0.5)
    
    return results

# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_results(atts_results: List[Dict], baseline_results: List[Dict]) -> Dict:
    """Analyze and compare results"""
    atts_df = pd.DataFrame(atts_results)
    baseline_df = pd.DataFrame(baseline_results)
    
    analysis = {
        "atts": {
            "accuracy": atts_df["correct"].mean() * 100,
            "avg_tokens": atts_df["tokens"].mean(),
            "total_tokens": atts_df["tokens"].sum(),
            "mode_distribution": atts_df["mode"].value_counts().to_dict(),
            "escalation_rate": atts_df["escalated"].mean() * 100 if "escalated" in atts_df else 0
        },
        "baseline": {
            "accuracy": baseline_df["correct"].mean() * 100,
            "avg_tokens": baseline_df["tokens"].mean(),
            "total_tokens": baseline_df["tokens"].sum()
        }
    }
    
    # Calculate savings
    analysis["comparison"] = {
        "token_savings_pct": (1 - analysis["atts"]["avg_tokens"] / analysis["baseline"]["avg_tokens"]) * 100,
        "accuracy_diff": analysis["atts"]["accuracy"] - analysis["baseline"]["accuracy"],
        "efficiency_ratio_atts": analysis["atts"]["accuracy"] / (analysis["atts"]["avg_tokens"] / 1000),
        "efficiency_ratio_baseline": analysis["baseline"]["accuracy"] / (analysis["baseline"]["avg_tokens"] / 1000)
    }
    
    # Difficulty estimation accuracy
    difficulty_map = {"easy": "direct", "medium": "thinking", "hard": "deep"}
    atts_df["expected_mode"] = atts_df["true_difficulty"].map(difficulty_map)
    analysis["difficulty_estimation"] = {
        "classification_accuracy": (atts_df["original_mode"] == atts_df["expected_mode"]).mean() * 100
    }
    
    return analysis

def print_report(analysis: Dict):
    """Print formatted analysis report"""
    print("\n" + "="*60)
    print("ATTS EXPERIMENT RESULTS")
    print("="*60)
    
    print("\n📊 BASELINE (Always Deep Mode)")
    print(f"   Accuracy: {analysis['baseline']['accuracy']:.1f}%")
    print(f"   Avg Tokens: {analysis['baseline']['avg_tokens']:.0f}")
    
    print("\n🎯 ATTS (Adaptive Mode)")
    print(f"   Accuracy: {analysis['atts']['accuracy']:.1f}%")
    print(f"   Avg Tokens: {analysis['atts']['avg_tokens']:.0f}")
    print(f"   Mode Distribution: {analysis['atts']['mode_distribution']}")
    print(f"   Escalation Rate: {analysis['atts']['escalation_rate']:.1f}%")
    
    print("\n📈 COMPARISON")
    print(f"   Token Savings: {analysis['comparison']['token_savings_pct']:.1f}%")
    print(f"   Accuracy Difference: {analysis['comparison']['accuracy_diff']:+.1f}%")
    print(f"   Efficiency (ATTS): {analysis['comparison']['efficiency_ratio_atts']:.2f} acc/%/1K tokens")
    print(f"   Efficiency (Baseline): {analysis['comparison']['efficiency_ratio_baseline']:.2f} acc/%/1K tokens")
    
    print("\n🎲 DIFFICULTY ESTIMATION")
    print(f"   Classification Accuracy: {analysis['difficulty_estimation']['classification_accuracy']:.1f}%")
    
    print("\n" + "="*60)
    
    # Verdict
    if analysis['comparison']['token_savings_pct'] > 20 and analysis['comparison']['accuracy_diff'] > -5:
        print("✅ HYPOTHESIS SUPPORTED: ATTS saves tokens while maintaining accuracy!")
    elif analysis['comparison']['token_savings_pct'] > 0:
        print("⚠️  PARTIAL SUPPORT: Some savings, but trade-offs exist.")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED: No significant savings observed.")
    
    print("="*60 + "\n")

def save_results(atts_results: List[Dict], baseline_results: List[Dict], analysis: Dict, output_dir: str = "."):
    """Save all results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    with open(f"{output_dir}/atts_results_{timestamp}.json", "w") as f:
        json.dump(atts_results, f, indent=2)
    
    with open(f"{output_dir}/baseline_results_{timestamp}.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    # Save analysis
    with open(f"{output_dir}/analysis_{timestamp}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Results saved with timestamp: {timestamp}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ATTS Simulation Experiment")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dataset", default="math_problems.json")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    
    print(f"\n🚀 Starting ATTS Experiment")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.dataset}\n")
    
    # Load dataset
    with open(args.dataset, "r") as f:
        data = json.load(f)
    problems = data["problems"]
    print(f"Loaded {len(problems)} problems\n")
    
    # Initialize client
    if args.provider == "openai":
        client = get_openai_client()
    else:
        client = get_anthropic_client()
    
    # Run experiments
    print("Running ATTS experiment...")
    atts_results = run_atts_experiment(client, args.provider, args.model, problems)
    
    print("\nRunning baseline experiment...")
    baseline_results = run_baseline_experiment(client, args.provider, args.model, problems)
    
    # Analyze
    analysis = analyze_results(atts_results, baseline_results)
    
    # Report
    print_report(analysis)
    
    # Save
    save_results(atts_results, baseline_results, analysis, args.output)

if __name__ == "__main__":
    main()
```

---

# 5. RUNNING THE EXPERIMENT STEP-BY-STEP

## Step 5.1: Verify Setup
```bash
cd atts_experiment
ls -la
# Should see: atts_experiment.py, math_problems.json

# Test API key
echo $OPENAI_API_KEY  # Should show your key (or part of it)
```

## Step 5.2: Run with OpenAI (Recommended for Lower Cost)
```bash
# Using GPT-4o-mini (cheapest, good for testing)
python atts_experiment.py --provider openai --model gpt-4o-mini

# Using GPT-4o (more capable, higher cost)
python atts_experiment.py --provider openai --model gpt-4o
```

## Step 5.3: Run with Anthropic (Alternative)
```bash
# Using Claude 3.5 Sonnet
python atts_experiment.py --provider anthropic --model claude-3-5-sonnet-20241022

# Using Claude 3.5 Haiku (cheaper)
python atts_experiment.py --provider anthropic --model claude-3-5-haiku-20241022
```

## Step 5.4: Expected Runtime
- 20 problems × 2 experiments × ~2 seconds per call = ~2 minutes
- With 100 problems: ~10 minutes

## Step 5.5: Monitor Progress
The script shows progress bars for each experiment phase.

---

# 6. ANALYZING RESULTS

## Step 6.1: Review Console Output
The script prints a formatted report including:
- Accuracy comparison
- Token savings percentage
- Mode distribution (how many Direct/Thinking/Deep)
- Difficulty estimation accuracy

## Step 6.2: Examine Saved Files
```bash
ls -la *.json
# atts_results_YYYYMMDD_HHMMSS.json - Raw ATTS results
# baseline_results_YYYYMMDD_HHMMSS.json - Raw baseline results  
# analysis_YYYYMMDD_HHMMSS.json - Computed analysis
```

## Step 6.3: Create Visualizations (Optional)
```python
# visualization.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load results
with open("analysis_YYYYMMDD_HHMMSS.json") as f:  # Use your timestamp
    analysis = json.load(f)

# Bar chart: Token comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Token usage
methods = ['Baseline (Always Deep)', 'ATTS (Adaptive)']
tokens = [analysis['baseline']['avg_tokens'], analysis['atts']['avg_tokens']]
axes[0].bar(methods, tokens, color=['#ff6b6b', '#4ecdc4'])
axes[0].set_ylabel('Average Tokens')
axes[0].set_title('Token Usage Comparison')

# Plot 2: Accuracy
accuracy = [analysis['baseline']['accuracy'], analysis['atts']['accuracy']]
axes[1].bar(methods, accuracy, color=['#ff6b6b', '#4ecdc4'])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Comparison')
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('atts_comparison.png', dpi=150)
plt.show()
```

---

# 7. TROUBLESHOOTING

## Error: "API key not found"
```bash
# Make sure key is exported
export OPENAI_API_KEY="sk-your-key"
# Or check .env file
```

## Error: "Rate limit exceeded"
- Increase `time.sleep(0.5)` to `time.sleep(1.0)` in the code
- Or use a model with higher rate limits

## Error: "Module not found"
```bash
pip install openai anthropic tiktoken pandas matplotlib tqdm
```

## Results seem off
- Check if answers in dataset are correct
- Verify the answer checking logic handles your format
- Try with a more capable model (gpt-4o instead of gpt-4o-mini)

---

# 8. EXPECTED RESULTS & INTERPRETATION

## What Success Looks Like
If ATTS hypothesis is valid:
- **Token Savings**: 30-50% reduction vs baseline
- **Accuracy**: Within 5% of baseline (or better)
- **Classification Accuracy**: >70% for difficulty estimation
- **Efficiency Ratio**: Higher for ATTS than baseline

## Example Good Results
```
BASELINE (Always Deep Mode)
   Accuracy: 85.0%
   Avg Tokens: 450

ATTS (Adaptive Mode)
   Accuracy: 82.5%
   Avg Tokens: 210
   Mode Distribution: {'direct': 8, 'thinking': 7, 'deep': 5}
   
COMPARISON
   Token Savings: 53.3%
   Accuracy Difference: -2.5%
   
✅ HYPOTHESIS SUPPORTED
```

## What to Do with Results

### If Successful:
1. Document your findings
2. Try with larger dataset (100+ problems)
3. Share on Twitter/Reddit with methodology
4. Write up as blog post or short paper

### If Mixed:
1. Tune thresholds (THRESHOLD_DIRECT, THRESHOLD_THINKING)
2. Try different models
3. Improve difficulty prompting
4. Add more escalation triggers

### If Failed:
1. Check for bugs in answer verification
2. Try more distinct difficulty levels
3. Consider that the model may already be efficient
4. Document negative results (still valuable!)

---

# 9. EXTENDING THE EXPERIMENT

## Ideas for Follow-up
1. **Different Domains**: Try coding problems, logic puzzles
2. **Escalation Variants**: Test different escalation thresholds
3. **Prompt Engineering**: Optimize the difficulty estimation prompt
4. **Model Comparison**: Compare GPT-4 vs Claude vs Gemini
5. **Larger Scale**: 500+ problems for statistical significance

## Publishing Your Results
If you get interesting results:
- **Twitter/X**: Post graphs + methodology
- **Reddit r/LocalLLaMA**: Share findings
- **arXiv**: Write up as short technical report
- **Blog**: Detailed walkthrough with code

---

# QUICK START CHECKLIST

□ Python installed
□ Packages installed (pip install openai anthropic tiktoken pandas matplotlib tqdm)
□ API key set (export OPENAI_API_KEY="...")
□ Dataset created (math_problems.json)
□ Script saved (atts_experiment.py)
□ Run: python atts_experiment.py --provider openai --model gpt-4o-mini
□ Review results
□ Share findings!

Good luck with your experiment! 🚀
