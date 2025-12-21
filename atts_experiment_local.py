"""
ATTS Simulation Experiment - LOCAL VERSION (Ollama)
100% Free, runs on your computer, no internet needed after setup!

Setup:
1. Install Ollama: https://ollama.ai/download
2. Pull a model: ollama pull llama3.2  (or mistral, phi3, etc.)
3. pip install ollama pandas tqdm
4. python atts_experiment_local.py --model llama3.2
"""

import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import re
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

THRESHOLD_DIRECT = 4
THRESHOLD_THINKING = 7
ESCALATION_THRESHOLD = 0.6

# ============================================================================
# PROMPTS
# ============================================================================

DIFFICULTY_PROMPT = """Rate the difficulty of this math problem from 1-10.
1-3: Easy, 4-6: Medium, 7-10: Hard
Output ONLY a number.

Problem: {problem}
Difficulty:"""

DIRECT_PROMPT = """Solve concisely: {problem}
Answer:"""

THINKING_PROMPT = """Solve step by step: {problem}
Steps:"""

DEEP_PROMPT = """Solve carefully with verification: {problem}
1. Understand the problem
2. Solve step by step  
3. Verify answer
Solution:"""

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

def call_ollama(model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call local Ollama model"""
    import ollama
    
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"num_predict": max_tokens, "temperature": 0.7}
        )
        text = response["response"]
        tokens = response.get("eval_count", len(text.split()))
        return text, tokens
    except Exception as e:
        print(f"Ollama Error: {e}")
        print("Make sure Ollama is running: ollama serve")
        return "", 0

# ============================================================================
# CORE LOGIC
# ============================================================================

def estimate_difficulty(model: str, problem: str) -> int:
    prompt = DIFFICULTY_PROMPT.format(problem=problem)
    response, _ = call_ollama(model, prompt, max_tokens=10)
    try:
        numbers = re.findall(r'\d+', response)
        if numbers:
            return max(1, min(10, int(numbers[0])))
    except:
        pass
    return 5

def select_mode(difficulty: int) -> str:
    if difficulty < THRESHOLD_DIRECT:
        return "direct"
    elif difficulty < THRESHOLD_THINKING:
        return "thinking"
    return "deep"

def solve_problem(model: str, problem: str, mode: str) -> Tuple[str, int]:
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 150
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 500
    else:
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 1000
    return call_ollama(model, prompt, max_tokens)

def check_answer(response: str, correct_answer: str) -> bool:
    response_lower = response.lower().replace(" ", "")
    answer_lower = correct_answer.lower().replace(" ", "")
    
    if answer_lower in response_lower:
        return True
    try:
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
# EXPERIMENTS
# ============================================================================

def run_atts_experiment(model: str, problems: List[Dict]) -> List[Dict]:
    results = []
    for prob in tqdm(problems, desc="ATTS"):
        difficulty = estimate_difficulty(model, prob["problem"])
        mode = select_mode(difficulty)
        solution, tokens = solve_problem(model, prob["problem"], mode)
        correct = check_answer(solution, prob["answer"])
        
        results.append({
            "id": prob["id"],
            "true_difficulty": prob["difficulty_label"],
            "predicted_difficulty": difficulty,
            "mode": mode,
            "tokens": tokens,
            "correct": correct
        })
    return results

def run_baseline_experiment(model: str, problems: List[Dict]) -> List[Dict]:
    results = []
    for prob in tqdm(problems, desc="Baseline"):
        solution, tokens = solve_problem(model, prob["problem"], "deep")
        correct = check_answer(solution, prob["answer"])
        results.append({
            "id": prob["id"],
            "mode": "deep",
            "tokens": tokens,
            "correct": correct
        })
    return results

def analyze_and_print(atts_results: List[Dict], baseline_results: List[Dict]):
    atts_df = pd.DataFrame(atts_results)
    baseline_df = pd.DataFrame(baseline_results)
    
    atts_acc = atts_df["correct"].mean() * 100
    atts_tokens = atts_df["tokens"].mean()
    baseline_acc = baseline_df["correct"].mean() * 100
    baseline_tokens = baseline_df["tokens"].mean()
    
    savings = (1 - atts_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0
    
    print("\n" + "="*50)
    print("📊 RESULTS (Local Ollama)")
    print("="*50)
    print(f"\nBaseline: {baseline_acc:.1f}% accuracy, {baseline_tokens:.0f} avg tokens")
    print(f"ATTS:     {atts_acc:.1f}% accuracy, {atts_tokens:.0f} avg tokens")
    print(f"\n💰 Token Savings: {savings:.1f}%")
    print(f"📈 Mode Distribution: {atts_df['mode'].value_counts().to_dict()}")
    print("="*50)
    
    if savings > 20 and (atts_acc - baseline_acc) > -5:
        print("✅ HYPOTHESIS SUPPORTED!")
    else:
        print("⚠️  Mixed results - try tuning thresholds")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--dataset", default="math_problems.json")
    args = parser.parse_args()
    
    print(f"\n🏠 ATTS Local Experiment (Ollama: {args.model})")
    print("="*50)
    
    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("✅ Ollama connected")
    except:
        print("❌ Ollama not running!")
        print("   Start it with: ollama serve")
        print("   Then pull a model: ollama pull llama3.2")
        return
    
    # Load problems
    with open(args.dataset) as f:
        problems = json.load(f)["problems"]
    print(f"Loaded {len(problems)} problems\n")
    
    # Run experiments
    atts_results = run_atts_experiment(args.model, problems)
    baseline_results = run_baseline_experiment(args.model, problems)
    
    # Report
    analyze_and_print(atts_results, baseline_results)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"local_results_{timestamp}.json", "w") as f:
        json.dump({"atts": atts_results, "baseline": baseline_results}, f, indent=2)
    print(f"\nSaved: local_results_{timestamp}.json")

if __name__ == "__main__":
    main()
