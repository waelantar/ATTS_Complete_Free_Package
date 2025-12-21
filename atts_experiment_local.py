"""
ATTS Experiment - Enhanced Local Version (Ollama)
Validates paper sections: 2.1 (USVA), 2.3 (ATTS), 2.3.3 (Escalation), 4.1 (Simulation Protocol)

Setup:
1. Install Docker and run Ollama: docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
2. Pull model: docker exec -it ollama ollama pull qwen2.5:3b-instruct
3. pip install -r requirements.txt
4. python atts_experiment_local.py --model qwen2.5:3b-instruct
"""

import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION (Section 2.3.2 - Compute Allocation Policy)
# ============================================================================

THRESHOLD_DIRECT = 4        # d < 4 → Direct
THRESHOLD_THINKING = 7      # 4 ≤ d < 7 → Thinking, d ≥ 7 → Deep
ESCALATION_THRESHOLD = 0.6  # Section 2.3.3 - verification score threshold

# ============================================================================
# PROMPTS (Based on Section 4.1)
# ============================================================================

DIFFICULTY_PROMPT = """Rate the difficulty of this math problem from 1-10.
1-3: Easy (single-step arithmetic)
4-6: Medium (multi-step reasoning)
7-10: Hard (competition-level, requires deep reasoning)
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

# Section 2.1.2 - USVA Generalized Verification Rubrics
VERIFICATION_PROMPT = """Evaluate this solution using these criteria (score 0-1 each):
1. Logical Coherence (LC): Do steps follow logically?
2. Factual Correctness (FC): Are calculations/facts correct?
3. Completeness (CM): Are all aspects addressed?
4. Goal Alignment (GA): Does it solve the stated problem?

Problem: {problem}
Solution: {solution}

Output format:
LC: [0-1 score]
FC: [0-1 score]
CM: [0-1 score]
GA: [0-1 score]
Overall: [average]"""

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

def call_ollama(model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call local Ollama model"""
    import ollama

    client = ollama.Client(host='http://localhost:11434')

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"num_predict": max_tokens, "temperature": 0.7}
        )
        text = response["response"]
        tokens = response.get("eval_count", len(text.split()))
        return text, tokens
    except Exception as e:
        print(f"Ollama Error: {e}")
        print("Make sure Ollama is running: docker start ollama")
        return "", 0

# ============================================================================
# CORE LOGIC - Section 2.3 (ATTS)
# ============================================================================

def estimate_difficulty(model: str, problem: str) -> int:
    """Section 2.3.1 - Difficulty Estimation"""
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
    """Section 2.3.2 - Mode Selection"""
    if difficulty < THRESHOLD_DIRECT:
        return "direct"
    elif difficulty < THRESHOLD_THINKING:
        return "thinking"
    return "deep"

def solve_problem(model: str, problem: str, mode: str) -> Tuple[str, int]:
    """Execute problem solving in specified mode"""
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 150
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 500
    else:  # deep
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 1000
    return call_ollama(model, prompt, max_tokens)

def verify_solution(model: str, problem: str, solution: str) -> float:
    """Section 2.1 - USVA-inspired Self-Verification

    Returns verification score v ∈ [0, 1] based on 4 rubrics:
    - Logical Coherence (LC)
    - Factual Correctness (FC)
    - Completeness (CM)
    - Goal Alignment (GA)
    """
    if not solution.strip():
        return 0.0

    prompt = VERIFICATION_PROMPT.format(problem=problem, solution=solution)
    response, _ = call_ollama(model, prompt, max_tokens=200)

    # Parse verification scores
    try:
        # Look for "Overall: X" or calculate average of LC, FC, CM, GA
        overall_match = re.search(r'Overall:\s*([\d.]+)', response)
        if overall_match:
            return float(overall_match.group(1))

        # Fallback: average individual scores
        scores = []
        for rubric in ['LC:', 'FC:', 'CM:', 'GA:']:
            match = re.search(rf'{rubric}\s*([\d.]+)', response)
            if match:
                scores.append(float(match.group(1)))

        if scores:
            return sum(scores) / len(scores)
    except:
        pass

    # Default: assume moderate confidence if can't parse
    return 0.5

def escalate_mode(current_mode: str) -> Optional[str]:
    """Section 2.3.3 - Escalation to next compute tier"""
    if current_mode == "direct":
        return "thinking"
    elif current_mode == "thinking":
        return "deep"
    return None  # Already at highest tier

def check_answer(response: str, correct_answer: str) -> bool:
    """Check if response contains correct answer"""
    response_lower = response.lower().replace(" ", "")
    answer_lower = correct_answer.lower().replace(" ", "")

    if answer_lower in response_lower:
        return True

    # Try numerical comparison
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
# EXPERIMENTS - Section 4.1 (Simulation Protocol)
# ============================================================================

def run_atts_experiment(model: str, problems: List[Dict], enable_escalation: bool = True) -> List[Dict]:
    """Run ATTS experiment with optional escalation"""
    results = []

    for prob in tqdm(problems, desc="ATTS"):
        # Stage 1: Difficulty Estimation
        difficulty = estimate_difficulty(model, prob["problem"])

        # Stage 2: Mode Selection
        mode = select_mode(difficulty)
        initial_mode = mode

        # Stage 3: Solution Generation
        solution, tokens = solve_problem(model, prob["problem"], mode)

        # Stage 4: Self-Verification (USVA)
        verification_score = verify_solution(model, prob["problem"], solution)

        # Stage 5: Escalation Check (Section 2.3.3)
        escalated = False
        escalation_path = [mode]

        if enable_escalation and verification_score < ESCALATION_THRESHOLD:
            next_mode = escalate_mode(mode)
            if next_mode:
                # Re-solve with higher compute
                escalated = True
                mode = next_mode
                escalation_path.append(mode)
                solution, new_tokens = solve_problem(model, prob["problem"], mode)
                tokens += new_tokens
                # Re-verify
                verification_score = verify_solution(model, prob["problem"], solution)

        # Output: Final solution + metrics
        correct = check_answer(solution, prob["answer"])

        results.append({
            "id": prob["id"],
            "true_difficulty": prob["difficulty_label"],
            "predicted_difficulty": difficulty,
            "initial_mode": initial_mode,
            "final_mode": mode,
            "escalated": escalated,
            "escalation_path": escalation_path,
            "verification_score": verification_score,
            "tokens": tokens,
            "correct": correct
        })

    return results

def run_baseline_experiment(model: str, problems: List[Dict]) -> List[Dict]:
    """Baseline: Always use Deep mode (Section 4.1)"""
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

# ============================================================================
# ENHANCED ANALYSIS - Section 3 (Theoretical Analysis)
# ============================================================================

def analyze_and_print(atts_results: List[Dict], baseline_results: List[Dict]):
    """Comprehensive analysis of results"""
    atts_df = pd.DataFrame(atts_results)
    baseline_df = pd.DataFrame(baseline_results)

    # Overall metrics
    atts_acc = atts_df["correct"].mean() * 100
    atts_tokens = atts_df["tokens"].mean()
    baseline_acc = baseline_df["correct"].mean() * 100
    baseline_tokens = baseline_df["tokens"].mean()

    savings = (1 - atts_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0

    print("\n" + "="*50)
    print("📊 RESULTS")
    print("="*50)
    print(f"\nBaseline: {baseline_acc:.1f}% accuracy, {baseline_tokens:.0f} avg tokens")
    print(f"ATTS:     {atts_acc:.1f}% accuracy, {atts_tokens:.0f} avg tokens")
    print(f"\n💰 Token Savings: {savings:.1f}%")

    # Mode distribution
    final_mode_dist = atts_df['final_mode'].value_counts().to_dict()
    print(f"📈 Mode Distribution: {final_mode_dist}")

    # Escalation statistics (Section 2.3.3)
    escalation_rate = atts_df["escalated"].mean() * 100
    print(f"🔼 Escalation Rate: {escalation_rate:.1f}%")

    # Difficulty classification accuracy
    difficulty_map = {"easy": 3, "medium": 5, "hard": 8}
    atts_df["true_difficulty_score"] = atts_df["true_difficulty"].map(difficulty_map)
    difficulty_mae = abs(atts_df["predicted_difficulty"] - atts_df["true_difficulty_score"]).mean()
    print(f"🎯 Difficulty Estimation MAE: {difficulty_mae:.2f}")

    # Per-difficulty breakdown
    print(f"\n📊 Performance by Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        atts_subset = atts_df[atts_df["true_difficulty"] == diff]
        baseline_subset = baseline_df[baseline_df["id"].isin(atts_subset["id"])]

        if len(atts_subset) > 0:
            atts_diff_acc = atts_subset["correct"].mean() * 100
            baseline_diff_acc = baseline_subset["correct"].mean() * 100
            atts_diff_tokens = atts_subset["tokens"].mean()
            baseline_diff_tokens = baseline_subset["tokens"].mean()
            diff_savings = (1 - atts_diff_tokens / baseline_diff_tokens) * 100 if baseline_diff_tokens > 0 else 0

            print(f"  {diff.capitalize()}: ATTS={atts_diff_acc:.0f}% / Baseline={baseline_diff_acc:.0f}% | "
                  f"Tokens: {atts_diff_tokens:.0f} vs {baseline_diff_tokens:.0f} ({diff_savings:+.1f}%)")

    # Verification score statistics
    avg_verification = atts_df["verification_score"].mean()
    print(f"\n✓ Average Verification Score: {avg_verification:.2f}")

    print("="*50)

    # Hypothesis validation (Section 4.2)
    if savings > 20 and (atts_acc - baseline_acc) > -5:
        print("✅ HYPOTHESIS SUPPORTED!")
        print("   • Token savings > 20%")
        print("   • Accuracy within 5% of baseline")
    else:
        print("⚠️  Mixed results - consider tuning thresholds")

    print("="*50)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATTS Experiment - Validates paper sections 2.1, 2.3, 4.1"
    )
    parser.add_argument("--model", default="qwen2.5:3b-instruct", help="Ollama model name")
    parser.add_argument("--dataset", default="data/math_problems.json", help="Dataset path")
    parser.add_argument("--no-escalation", action="store_true", help="Disable escalation mechanism")
    parser.add_argument("--threshold", type=float, default=0.6, help="Verification threshold for escalation")
    args = parser.parse_args()

    global ESCALATION_THRESHOLD
    ESCALATION_THRESHOLD = args.threshold

    print(f"\n🏠 ATTS Local Experiment (Ollama: {args.model})")
    print("="*50)

    # Check Ollama
    try:
        import ollama
        client = ollama.Client(host='http://localhost:11434')
        client.list()
        print("✅ Ollama connected")
    except Exception as e:
        print(f"❌ Ollama not running! Error: {e}")
        print("   Start: docker start ollama")
        print("   Pull model: docker exec -it ollama ollama pull qwen2.5:3b-instruct")
        return

    # Load problems
    try:
        with open(args.dataset) as f:
            problems = json.load(f)["problems"]
        print(f"📂 Loaded {len(problems)} problems")
    except FileNotFoundError:
        print(f"❌ Dataset not found: {args.dataset}")
        return

    print(f"\n⚙️  Configuration:")
    print(f"   • Difficulty thresholds: Direct<{THRESHOLD_DIRECT}, Thinking<{THRESHOLD_THINKING}")
    print(f"   • Escalation: {'Enabled' if not args.no_escalation else 'Disabled'}")
    print(f"   • Verification threshold: {ESCALATION_THRESHOLD}")
    print()

    # Run experiments
    atts_results = run_atts_experiment(args.model, problems, enable_escalation=not args.no_escalation)
    baseline_results = run_baseline_experiment(args.model, problems)

    # Analyze
    analyze_and_print(atts_results, baseline_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/local_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "threshold_direct": THRESHOLD_DIRECT,
                "threshold_thinking": THRESHOLD_THINKING,
                "escalation_threshold": ESCALATION_THRESHOLD,
                "escalation_enabled": not args.no_escalation
            },
            "atts": atts_results,
            "baseline": baseline_results
        }, f, indent=2)
    print(f"\n💾 Saved: {output_file}")

if __name__ == "__main__":
    main()
