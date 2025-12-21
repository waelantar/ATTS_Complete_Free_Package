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
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

THRESHOLD_DIRECT = 4      # d < 4 -> Direct mode
THRESHOLD_THINKING = 7    # 4 <= d < 7 -> Thinking mode
ESCALATION_THRESHOLD = 0.6

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
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    tokens = response.usage.output_tokens
    return text, tokens

def call_model(client, provider: str, model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
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
    prompt = DIFFICULTY_PROMPT.format(problem=problem)
    response, _ = call_model(client, provider, model, prompt, max_tokens=10)
    try:
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            difficulty = int(numbers[0])
            return max(1, min(10, difficulty))
    except:
        pass
    return 5

def select_mode(difficulty: int) -> str:
    if difficulty < THRESHOLD_DIRECT:
        return "direct"
    elif difficulty < THRESHOLD_THINKING:
        return "thinking"
    else:
        return "deep"

def solve_problem(client, provider: str, model: str, problem: str, mode: str) -> Tuple[str, int]:
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 200
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 800
    else:
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 2000
    return call_model(client, provider, model, prompt, max_tokens)

def verify_solution(client, provider: str, model: str, problem: str, solution: str) -> float:
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
    return 0.7

def check_answer(response: str, correct_answer: str) -> bool:
    response_lower = response.lower().replace(" ", "")
    answer_lower = correct_answer.lower().replace(" ", "")
    
    if answer_lower in response_lower:
        return True
    
    try:
        import re
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
    results = []
    for prob in tqdm(problems, desc="ATTS Experiment"):
        difficulty = estimate_difficulty(client, provider, model, prob["problem"])
        mode = select_mode(difficulty)
        original_mode = mode
        
        solution, tokens = solve_problem(client, provider, model, prob["problem"], mode)
        total_tokens = tokens
        
        escalated = False
        if mode in ["thinking", "deep"]:
            confidence = verify_solution(client, provider, model, prob["problem"], solution)
            if confidence < ESCALATION_THRESHOLD and mode != "deep":
                escalated = True
                mode = "deep"
                solution, extra_tokens = solve_problem(client, provider, model, prob["problem"], mode)
                total_tokens += extra_tokens
        
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
            "solution": solution[:500]
        })
        time.sleep(0.5)
    return results

def run_baseline_experiment(client, provider: str, model: str, problems: List[Dict]) -> List[Dict]:
    results = []
    for prob in tqdm(problems, desc="Baseline Experiment"):
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
# ANALYSIS
# ============================================================================

def analyze_results(atts_results: List[Dict], baseline_results: List[Dict]) -> Dict:
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
    
    analysis["comparison"] = {
        "token_savings_pct": (1 - analysis["atts"]["avg_tokens"] / analysis["baseline"]["avg_tokens"]) * 100,
        "accuracy_diff": analysis["atts"]["accuracy"] - analysis["baseline"]["accuracy"],
        "efficiency_ratio_atts": analysis["atts"]["accuracy"] / (analysis["atts"]["avg_tokens"] / 1000),
        "efficiency_ratio_baseline": analysis["baseline"]["accuracy"] / (analysis["baseline"]["avg_tokens"] / 1000)
    }
    
    difficulty_map = {"easy": "direct", "medium": "thinking", "hard": "deep"}
    atts_df["expected_mode"] = atts_df["true_difficulty"].map(difficulty_map)
    analysis["difficulty_estimation"] = {
        "classification_accuracy": (atts_df["original_mode"] == atts_df["expected_mode"]).mean() * 100
    }
    
    return analysis

def print_report(analysis: Dict):
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
    
    if analysis['comparison']['token_savings_pct'] > 20 and analysis['comparison']['accuracy_diff'] > -5:
        print("✅ HYPOTHESIS SUPPORTED: ATTS saves tokens while maintaining accuracy!")
    elif analysis['comparison']['token_savings_pct'] > 0:
        print("⚠️  PARTIAL SUPPORT: Some savings, but trade-offs exist.")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED: No significant savings observed.")
    print("="*60 + "\n")

def save_results(atts_results: List[Dict], baseline_results: List[Dict], analysis: Dict, output_dir: str = "."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/atts_results_{timestamp}.json", "w") as f:
        json.dump(atts_results, f, indent=2)
    
    with open(f"{output_dir}/baseline_results_{timestamp}.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
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
    
    with open(args.dataset, "r") as f:
        data = json.load(f)
    problems = data["problems"]
    print(f"Loaded {len(problems)} problems\n")
    
    if args.provider == "openai":
        client = get_openai_client()
    else:
        client = get_anthropic_client()
    
    print("Running ATTS experiment...")
    atts_results = run_atts_experiment(client, args.provider, args.model, problems)
    
    print("\nRunning baseline experiment...")
    baseline_results = run_baseline_experiment(client, args.provider, args.model, problems)
    
    analysis = analyze_results(atts_results, baseline_results)
    print_report(analysis)
    save_results(atts_results, baseline_results, analysis, args.output)

if __name__ == "__main__":
    main()
