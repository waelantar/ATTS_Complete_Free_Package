"""
ATTS Simulation Experiment - FREE VERSION (Google Gemini)
No credit card required!

Setup:
1. Go to https://aistudio.google.com/app/apikey
2. Create a free API key
3. export GOOGLE_API_KEY="your-key"
4. pip install google-generativeai pandas tqdm
5. python atts_experiment_free.py
"""

import os
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

DIFFICULTY_PROMPT = """Rate the difficulty of this math problem on a scale of 1-10.
- 1-3: Very easy (single step, basic arithmetic)
- 4-6: Medium (multiple steps, requires some thought)
- 7-10: Hard (complex reasoning, multiple concepts)

IMPORTANT: Only output a single number from 1-10. Nothing else.

Problem: {problem}

Difficulty (1-10):"""

DIRECT_PROMPT = """Solve this math problem. Be concise.
Problem: {problem}
Answer:"""

THINKING_PROMPT = """Solve this math problem step by step.
Problem: {problem}
Let me work through this:"""

DEEP_PROMPT = """Solve this math problem carefully with thorough reasoning.
Problem: {problem}

I'll solve this systematically:
1. Understand the problem
2. Identify key concepts
3. Work through step by step
4. Verify my answer

Solution:"""

VERIFICATION_PROMPT = """Rate confidence in this solution from 0-10.
Problem: {problem}
Solution: {solution}
Confidence (0-10):"""

# ============================================================================
# GEMINI CLIENT
# ============================================================================

def get_gemini_model():
    """Initialize Gemini model"""
    import google.generativeai as genai
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 1.5 Flash (fast and free)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def call_gemini(model, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call Gemini API and return (response, estimated_tokens)"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.7
            }
        )
        text = response.text
        # Estimate tokens (Gemini doesn't always return exact count)
        estimated_tokens = len(text.split()) * 1.3  # Rough estimate
        return text, int(estimated_tokens)
    except Exception as e:
        print(f"API Error: {e}")
        return "", 0

# ============================================================================
# CORE ATTS LOGIC
# ============================================================================

def estimate_difficulty(model, problem: str) -> int:
    prompt = DIFFICULTY_PROMPT.format(problem=problem)
    response, _ = call_gemini(model, prompt, max_tokens=10)
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

def solve_problem(model, problem: str, mode: str) -> Tuple[str, int]:
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 200
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 800
    else:
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 2000
    return call_gemini(model, prompt, max_tokens)

def verify_solution(model, problem: str, solution: str) -> float:
    prompt = VERIFICATION_PROMPT.format(problem=problem, solution=solution)
    response, _ = call_gemini(model, prompt, max_tokens=10)
    try:
        numbers = re.findall(r'\d+', response)
        if numbers:
            return max(0, min(1, int(numbers[0]) / 10.0))
    except:
        pass
    return 0.7

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
# EXPERIMENT RUNNERS
# ============================================================================

def run_atts_experiment(model, problems: List[Dict]) -> List[Dict]:
    results = []
    for prob in tqdm(problems, desc="ATTS Experiment"):
        difficulty = estimate_difficulty(model, prob["problem"])
        mode = select_mode(difficulty)
        original_mode = mode
        
        solution, tokens = solve_problem(model, prob["problem"], mode)
        total_tokens = tokens
        
        escalated = False
        if mode in ["thinking", "deep"]:
            confidence = verify_solution(model, prob["problem"], solution)
            if confidence < ESCALATION_THRESHOLD and mode != "deep":
                escalated = True
                mode = "deep"
                solution, extra_tokens = solve_problem(model, prob["problem"], mode)
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
        time.sleep(1)  # Rate limiting for free tier
    return results

def run_baseline_experiment(model, problems: List[Dict]) -> List[Dict]:
    results = []
    for prob in tqdm(problems, desc="Baseline (Always Deep)"):
        solution, tokens = solve_problem(model, prob["problem"], "deep")
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
        time.sleep(1)
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
            "escalation_rate": atts_df["escalated"].mean() * 100
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
    }
    
    difficulty_map = {"easy": "direct", "medium": "thinking", "hard": "deep"}
    atts_df["expected_mode"] = atts_df["true_difficulty"].map(difficulty_map)
    analysis["difficulty_estimation"] = {
        "classification_accuracy": (atts_df["original_mode"] == atts_df["expected_mode"]).mean() * 100
    }
    
    return analysis

def print_report(analysis: Dict):
    print("\n" + "="*60)
    print("🧪 ATTS EXPERIMENT RESULTS (Gemini Free Tier)")
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
    
    print("\n🎲 DIFFICULTY ESTIMATION")
    print(f"   Classification Accuracy: {analysis['difficulty_estimation']['classification_accuracy']:.1f}%")
    
    print("\n" + "="*60)
    if analysis['comparison']['token_savings_pct'] > 20 and analysis['comparison']['accuracy_diff'] > -5:
        print("✅ HYPOTHESIS SUPPORTED!")
    elif analysis['comparison']['token_savings_pct'] > 0:
        print("⚠️  PARTIAL SUPPORT")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED")
    print("="*60 + "\n")

def save_results(atts_results, baseline_results, analysis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results_{timestamp}.json", "w") as f:
        json.dump({"atts": atts_results, "baseline": baseline_results, "analysis": analysis}, f, indent=2)
    print(f"Results saved: results_{timestamp}.json")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math_problems.json")
    args = parser.parse_args()
    
    print("\n🚀 ATTS Experiment (FREE - Google Gemini)")
    print("="*50)
    
    # Load problems
    with open(args.dataset) as f:
        problems = json.load(f)["problems"]
    print(f"Loaded {len(problems)} problems\n")
    
    # Initialize Gemini
    print("Connecting to Gemini API...")
    model = get_gemini_model()
    print("Connected!\n")
    
    # Run experiments
    print("Running ATTS experiment...")
    atts_results = run_atts_experiment(model, problems)
    
    print("\nRunning baseline experiment...")
    baseline_results = run_baseline_experiment(model, problems)
    
    # Analyze and report
    analysis = analyze_results(atts_results, baseline_results)
    print_report(analysis)
    save_results(atts_results, baseline_results, analysis)

if __name__ == "__main__":
    main()
